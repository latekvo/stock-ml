import torch
from torch.utils.data import DataLoader
from model import Net
from dataset import DummyDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

ds = DummyDataset()
dl = DataLoader(ds, batch_size=32, shuffle=True)

model = Net(10, 32, 2).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(999):
    for x, y in dl:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(epoch, loss.item())