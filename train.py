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


def train():
    ds = FinancialDataset(INPUT_CANDLES, OUTPUT_CANDLES)

    x = ds.x.to(device, non_blocking=True)
    y = ds.y.to(device, non_blocking=True)

    model = Net(ds.net_inputs, 128, ds.net_outputs).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.015,
        foreach=True,
    )

    if device_name == "cuda":
        model = torch.compile(model)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    while True:
        out = model(x)
        loss = loss_fn(out, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        telemetry_registry.log_loss(loss.item())


threading.Thread(target=train, daemon=True).start()

run_visualisation_app(telemetry_registry)
