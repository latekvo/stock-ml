from typing import TypedDict, Literal

import torch
from model import Net
from dataset import FinancialDataset
import threading

from src.ui.app import run_visualisation_app
from src.core.telemetry import TelemetryRegistry

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

print("DEVICE:", device)

telemetry_registry = TelemetryRegistry()

COLORS = {
    'r': (255, 0, 0, 255),
    'g': (0, 255, 0, 255),
    'b': (0, 0, 255, 255),
    'c': (0, 255, 255, 255),
    'm': (255, 0, 255, 255),
    'y': (255, 255, 0, 255),
}

INPUT_CANDLES = 7
OUTPUT_CANDLES = 3

class ModelVariant(TypedDict):
    name: str
    p_per_layer: int
    layers: int
    learning_rate: float
    color: tuple[int, int, int, int]
    a_fn: Literal["ReLU", "GELU"]

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
# - 5e-2 (50x baseline) LR **significantly** improves train time without degrading loss. Sticking to 1e-2 for future tests.
# - 24/7 with 50x LR was extremely chaotic but ultimately outperformed 32/8
# - GELU, 24/7(50*lr), 32/8 all achieved 10.7673 within 1500 epochs, baseline at epoch 2100 - ended training early.

baseline = ModelVariant(name='baseline', p_per_layer=24, layers=7, learning_rate=1e-3, color=COLORS['m'], a_fn="ReLU")
gelu = ModelVariant(name='gelu', p_per_layer=24, layers=7, learning_rate=1e-3, color=COLORS['r'], a_fn="GELU")
high_lr = ModelVariant(name='high_lr', p_per_layer=24, layers=7, learning_rate=5e-2, color=COLORS['c'], a_fn="ReLU")
deep_large = ModelVariant(name='deep_large', p_per_layer=32, layers=8, learning_rate=1e-3, color=COLORS['y'], a_fn="ReLU")  # keeping old for reference

MODEL_VARIANTS: list[ModelVariant] = [
    baseline,
    gelu,
    high_lr,
    deep_large
]

def train():
    ds = FinancialDataset(INPUT_CANDLES, OUTPUT_CANDLES)

    t_x = ds.t_x.to(device, non_blocking=True)
    t_y = ds.t_y.to(device, non_blocking=True)
    e_x = ds.e_x.to(device, non_blocking=True)
    e_y = ds.e_y.to(device, non_blocking=True)

    models = {}
    opts = {}

    for variant in MODEL_VARIANTS:
        name = variant['name']

        models[name] = Net(
            ds.net_inputs,
            variant['p_per_layer'],
            variant['layers'],
            variant['a_fn'],
            ds.net_outputs
        ).to(device)

        opts[name] = torch.optim.AdamW(
            models[name].parameters(),
            lr=variant['learning_rate'],
            foreach=True,
        )

    loss_fn = torch.nn.CrossEntropyLoss()

    if device_name == "cuda":
        # requires modern GPU: model = torch.compile(model)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    while True:
        for variant in MODEL_VARIANTS:
            name = variant['name']
            model = models[name]
            opt = opts[name]

            out = model(t_x)
            eval_out = model(e_x)

            loss = loss_fn(out, t_y)
            eval_loss = loss_fn(eval_out, e_y)

            telemetry_registry.log_loss(str(name) + '_training', loss.item())
            telemetry_registry.log_loss(str(name) + '_eval', eval_loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()


for variant in MODEL_VARIANTS:
    name = variant['name']
    e_color = variant['color']
    color = list(e_color)
    color[3] = 128
    t_color = tuple(color)

    telemetry_registry.add_loss_entry(name + '_training', t_color)
    telemetry_registry.add_loss_entry(name + '_eval', e_color)

threading.Thread(target=train, daemon=True).start()

run_visualisation_app(telemetry_registry)
