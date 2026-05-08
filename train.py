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

    t_x = ds.t_x.to(device, non_blocking=True)
    t_y = ds.t_y.to(device, non_blocking=True)
    e_x = ds.e_x.to(device, non_blocking=True)
    e_y = ds.e_y.to(device, non_blocking=True)

    model = Net(ds.net_inputs, 128, ds.net_outputs).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.015,
        foreach=True,
    )

    if device_name == "cuda":
        # requires modern GPU: model = torch.compile(model)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    while True:
        out = model(t_x)
        loss = loss_fn(out, t_y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        eval_out = model(e_x)
        eval_loss = loss_fn(eval_out, e_y)

        telemetry_registry.log_loss('training', loss.item())
        telemetry_registry.log_loss('eval', eval_loss.item())


telemetry_registry.add_loss_entry('training', 'r')
telemetry_registry.add_loss_entry('eval', 'w')

threading.Thread(target=train, daemon=True).start()

run_visualisation_app(telemetry_registry)
