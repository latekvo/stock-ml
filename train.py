from typing import TypedDict, Literal

import torch
from torch.optim import lr_scheduler

from model import Net, Act
from dataset import FinancialDataset
import threading

from src.ui.app import run_visualisation_app
from src.core.telemetry import TelemetryRegistry

device_name = "cpu"

if torch.cuda.is_available():
    device_name = "cuda"
if torch.mps.is_available():
    device_name = "mps"

device = torch.device(device_name)

print("DEVICE:", device)

telemetry_registry = TelemetryRegistry()

COLORS = {
    "w": (255, 255, 255, 255),
    "r": (255, 0, 0, 255),
    "g": (0, 255, 0, 255),
    "b": (0, 0, 255, 255),
    "c": (0, 255, 255, 255),
    "m": (255, 0, 255, 255),
    "y": (255, 255, 0, 255),
}

INPUT_CANDLES = 7
OUTPUT_CANDLES = 3


class ModelVariant(TypedDict):
    name: str
    p_per_layer: int
    layers: int
    lr: float
    wd: float
    color: tuple[int, int, int, int]
    act: Act
    residual: bool
    normalize: bool
    optimizer: Literal["LBFGS", "AdamW"]
    lr_scheduler_gamma: float
    lr_scheduler_step: int
    lr_scheduler: Literal["step", "on_plateau"]


# 1st benchmark results:
# - LR noise seems to be proportional to: (lr*p_per_layer)/layers
# - Higher total param counts increase effective LR
# - few-layer models perform poorly, but are better at generalization
# - deep-models despite higher memorization rates perform substantially better
# - p_per_layer seems to have very little effect on the model's loss
# - best achieved loss was 10.95 eval CrossEntropy with 64/3 model.
# 1st benchmark conditions:
# - const 7/3 i/o with fib offsets + candle merging
# - net used 64/2 baseline, */2 p_per_layer variance, +-1 layers variance. LR constant 1e-4
# - ran tests for 2 hrs (~1200 epochs) on M4 Pro
# 1st benchmark followups:
# - try to achieve grokking with 32/5 model, then test how LR and WD affect the timeline
# - find optimal p_per_layer & layers at 1e-3 LR.
# =====================
# 2nd benchmark results:
# - 32/4 model took SIGNIFICANTLY longer to train
# - 32/8 model had almost no advantage over 32/6 - 0.00015 loss difference (0.00037% CE loss difference)
# - 32/6 model had almost no advantage over 16/6
# - 64/6 model was noticeably worse than all of the 6-8 layer models
# - no observable memorization rate differences between the models
# - best achieved loss was 10.767 eval CrossEntropy with 32/8 model.
# 2nd benchmark conditions:
# - const 7/3 i/o with fib offsets + candle merging
# - net used 32/6 baseline, */2 p_per_layer variance, +-2 layers variance. LR constant 1e-3
# - ran tests for 1.5 hr (~1100 epochs) on M4 Pro
# 2nd benchmark followups:
# - test 7 vs 8 layers, try 24 vs 32 p_per_layer
# - compare GELU vs ReLU
# - compare even higher LR
# =====================
# 3rd benchmark results:
# - GELU performs significatnly better than ReLU. Loss is unfortunately inconsistant.
# - 5e-2 (50x baseline) LR **significantly** improves initial train time, but degrades after hours. Sticking to 1e-2 for short runs.
# - 24/7 with 50x LR was extremely chaotic but ultimately outperformed 32/8
# - GELU, 24/7(50*lr), 32/8 all achieved 10.7673 within 1500 epochs, baseline projected to converge at epoch 2000 - ended training early.
# 3rd benchmark conditions:
# - const 7/3 i/o with fib offsets + candle merging
# - net used 24/7 baseline, +8 p_per_layer variance, +1 layers variance. LR constant 1e-3
# - ran tests for 1.5 hr (~1500 epochs) on M4 Pro
# 3rd benchmark followups:
# - test residual connections
# - test more activation functions
# - test much higher WD
# =====================
# 4th benchmark results:
# - All tested variables provided marginal gains, all current optimizations yield 0.001% differences, so usefulness of CE-based comparison is exhausted.
#   This means we have to move to market-simulation as the primary success metric
# - 1e-2 LR is excessive and not worth the jitter. Even 1e-3 is excessive later on - we need to use LR scheduler.
# - Current approach to residual connections experiences high jitter and converges incredibly slowly - the current implementation must be wrong
# - Achieved 10.76724: bearly a 0.003% improvement, possibly due to long runtime (~1900 epochs)
# 4th benchmark conclusions:
# - SiLU vs GELU: They behave identically, but vary in jitter profile - GELU consistantly gets gigantic loss spikes. SiLU is always jittery in a predictable oscillating way.
# - High WD seems to converge better, though it's within the margin of error. Keep it at 8e-2 for marginal gains.
# - Residual connections are not usable with current implementation, requires work.
# - Normalization appears worse, seems to amplify loss jitter, but within the margin of error.
# =====================
# minor benchmarks:
# - AdamW vs LBFGS: AdamW converges faster and to lower values. LBFGS is much less noisy.
# - Normalization significantly slows down large networks, likely due to graph breaks (code issue, not pytorch issue)
# - Increasing layers & parameters sets the loss floor higher (bad) due to unknown reasons (tested on 25k epochs, no difference)
# - absolute achievably floor with the current architecture seems to be 10.7671
# - LR Scheduling dramatically increases stability and steady learning rate at final stages of learning.
# - on_plateau LR scheduling does not work well with default settings. Too little value to tune it at this point in time

baseline: ModelVariant = {
    "name": "baseline",
    "p_per_layer": 24,
    "layers": 8,
    "lr": 3e-3,
    "wd": 8e-2,
    "color": COLORS["w"],
    "act": "SiLU",
    "residual": False,
    "normalize": False,
    "optimizer": "AdamW",
    "lr_scheduler_gamma": 0.989,  # ~10% LR every 2k epochs at step=10
    "lr_scheduler_step": 10,
    "lr_scheduler": "step",
}


on_plateau = {
    **baseline,
    "name": "on_plateau",
    "color": COLORS["c"],
    "lr_scheduler": "on_plateau",
}


MODEL_VARIANTS: list[ModelVariant] = [
    baseline,
    on_plateau,
]


def train():
    ds = FinancialDataset(INPUT_CANDLES, OUTPUT_CANDLES)

    t_x = ds.t_x.to(device, non_blocking=True)
    t_y = ds.t_y.to(device, non_blocking=True)
    e_x = ds.e_x.to(device, non_blocking=True)
    e_y = ds.e_y.to(device, non_blocking=True)

    models = {}
    opts = {}
    lr_schedulers = {}

    for variant in MODEL_VARIANTS:
        name = variant["name"]

        models[name] = Net(
            ds.net_inputs,
            ds.net_outputs,
            variant["p_per_layer"],
            variant["layers"],
            variant["act"],
            variant["residual"],
            variant["normalize"],
        ).to(device)

        if variant["optimizer"] == "LBFGS":
            opts[name] = torch.optim.LBFGS(
                models[name].parameters(),
                lr=variant["lr"],
            )
        elif variant["optimizer"] == "AdamW":
            opts[name] = torch.optim.AdamW(
                models[name].parameters(),
                lr=variant["lr"],
                weight_decay=variant["wd"],
                foreach=True,
            )
        else:
            raise ValueError("Invalid `optimizer` value")

        if variant["lr_scheduler"] == "on_plateau":
            lr_schedulers[name] = lr_scheduler.ReduceLROnPlateau(opts[name], "min")
        elif variant["lr_scheduler"] == "step":
            lr_schedulers[name] = lr_scheduler.StepLR(
                opts[name],
                step_size=variant["lr_scheduler_step"],
                gamma=variant["lr_scheduler_gamma"],
            )
        else:
            raise ValueError("Invalid `lr_scheduler` value")

    loss_fn = torch.nn.CrossEntropyLoss()

    if device_name == "cuda":
        # requires modern GPU: model = torch.compile(model)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    while True:
        for variant in MODEL_VARIANTS:
            name = variant["name"]
            model = models[name]
            opt = opts[name]

            eval_out = model(e_x)
            eval_loss = loss_fn(eval_out, e_y)

            def calc_loss():
                out = model(t_x)
                loss = loss_fn(out, t_y)
                opt.zero_grad()
                loss.backward()
                return loss

            loss = calc_loss()

            if variant["optimizer"] == "LBFGS":
                opt.step(lambda: loss)
            else:
                opt.step()

            if variant["lr_scheduler"] == "on_plateau":
                lr_schedulers[name].step(loss)
            elif variant["lr_scheduler"] == "step":
                lr_schedulers[name].step()

            telemetry_registry.log_loss(str(name) + "_training", loss.item())
            telemetry_registry.log_loss(str(name) + "_eval", eval_loss.item())


for variant in MODEL_VARIANTS:
    name = variant["name"]
    e_color = variant["color"]
    color = list(e_color)
    color[3] = 128
    t_color = tuple(color)

    telemetry_registry.add_loss_entry(name + "_training", t_color)
    telemetry_registry.add_loss_entry(name + "_eval", e_color)

threading.Thread(target=train, daemon=True).start()

run_visualisation_app(telemetry_registry)
