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

# class ModelSettings:
#     params_per_layer: int
#     layers: int
#     learning_rate: float

MODEL_VARIANTS = [
    64,
    128,
    512,
]

COLORS = {
    64: (0, 0, 255, 255),
    128: (255, 0, 0, 255),
    512: (0, 255, 0, 255),
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
        models[variant] = Net(ds.net_inputs, variant, ds.net_outputs).to(device)
        opts[variant] = torch.optim.AdamW(
            models[variant].parameters(),
            lr=1e-4,
            weight_decay=0.015,
            foreach=True,
        )

    loss_fn = torch.nn.CrossEntropyLoss()

    if device_name == "cuda":
        # requires modern GPU: model = torch.compile(model)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    while True:
        for variant in MODEL_VARIANTS:
            model = models[variant]
            opt = opts[variant]

            out = model(t_x)
            loss = loss_fn(out, t_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            eval_out = model(e_x)
            eval_loss = loss_fn(eval_out, e_y)

            telemetry_registry.log_loss(str(variant) + '_training', loss.item())
            telemetry_registry.log_loss(str(variant) + '_eval', eval_loss.item())


for variant in MODEL_VARIANTS:
    e_color = COLORS[variant]
    color = list(e_color)
    color[3] = 128
    t_color = tuple(color)

    telemetry_registry.add_loss_entry(str(variant) + '_training', t_color)
    telemetry_registry.add_loss_entry(str(variant) + '_eval', e_color)

threading.Thread(target=train, daemon=True).start()

run_visualisation_app(telemetry_registry)
