from typing import TypedDict

import torch
from torch.utils.data import DataLoader
from model import Net
from dataset import FinancialDataset
import threading

from src.ui.app import run_visualisation_app
from src.core.telemetry import TelemetryRegistry

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

print("DEVICE:", device)

telemetry_registry = TelemetryRegistry()


INPUT_CANDLES = 7
OUTPUT_CANDLES = 3

class ModelVariant(TypedDict):
    name: str
    p_per_layer: int
    layers: int
    learning_rate: float

# 1st benchmark results:
# - LR noise seems to be proportional to: (lr*p_per_layer)/layers
# - Higher total param counts increase effective LR
# - few-layer models perform poorly, but are better at generalization
# - deep-models despite higher memorization rates perform substantially better
# - p_per_layer seems to have very little effect on the model's loss
# - best achieved loss was 10.96 CrossEntropy with 64/3 model.
# 1st benchmark variants:
# - 64/2 baseline, */2 p_per_layer variance, +-1 layers variance. LR constant 1e-4
# 1st followups:
# - try to achieve grokking with 32/5 model, then test how LR and WD affect the timeline
# - find optimal p_per_layer & layers at 1e-3 LR.

baseline = ModelVariant(name='baseline', p_per_layer=64, layers=2, learning_rate=1e-4)
small = ModelVariant(name='small', p_per_layer=32, layers=2, learning_rate=1e-4)
large = ModelVariant(name='large', p_per_layer=128, layers=2, learning_rate=1e-4)
shallow = ModelVariant(name='shallow', p_per_layer=64, layers=1, learning_rate=1e-4)
deep = ModelVariant(name='deep', p_per_layer=64, layers=3, learning_rate=1e-4)

MODEL_VARIANTS: list[ModelVariant] = [
    baseline,
    small,
    large,
    shallow,
    deep,
]

COLORS = {
    'baseline': (255, 0, 255, 255),
    'small': (255, 0, 0, 255),
    'large': (0, 255, 0, 255),
    'shallow': (0, 255, 255, 255),
    'deep': (255, 255, 0, 255),
}

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
        models[name] = Net(ds.net_inputs, variant['p_per_layer'], variant['layers'], ds.net_outputs).to(device)
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
            loss = loss_fn(out, t_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            eval_out = model(e_x)
            eval_loss = loss_fn(eval_out, e_y)

            telemetry_registry.log_loss(str(name) + '_training', loss.item())
            telemetry_registry.log_loss(str(name) + '_eval', eval_loss.item())


for variant in MODEL_VARIANTS:
    name = variant['name']
    e_color = COLORS[name]
    color = list(e_color)
    color[3] = 128
    t_color = tuple(color)

    telemetry_registry.add_loss_entry(name + '_training', t_color)
    telemetry_registry.add_loss_entry(name + '_eval', e_color)

threading.Thread(target=train, daemon=True).start()

run_visualisation_app(telemetry_registry)
