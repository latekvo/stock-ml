import torch
from torch.utils.data import DataLoader
from model import Net
from dataset import FinancialDataset
import threading

from src.ui.app import run_visualisation_app
from src.core.telemetry import TelemetryRegistry

device = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:", device)

EPOCHS = 1000
BATCH_SIZE = 1000

telemetry_registry = TelemetryRegistry()


def train():
    ds = FinancialDataset(5, 2)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = Net(10, 32, 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        for x, y in dl:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = loss_fn(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        telemetry_registry.log_loss(loss.item())


threading.Thread(target=train, daemon=True).start()

run_visualisation_app(telemetry_registry)
