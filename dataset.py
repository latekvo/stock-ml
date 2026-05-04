import os
from pathlib import Path

from torch.utils.data import Dataset
import torch

from src.core.serialization import serialize_stooq_file


class DummyDataset(Dataset):
    def __init__(self, n=1000):
        self.x = torch.randn(n, 10)
        self.y = (self.x.sum(dim=1) > 0).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


DATA_RAW = os.path.join("data", "daily", "pl", "nc stocks")

files = list(Path(DATA_RAW).iterdir())


class FinancialDataset(Dataset):
    def __init__(self, n=1000):
        for file in files:
            filename = file.absolute().as_posix()
            candles = serialize_stooq_file(filename)
            print(file.absolute().as_posix(), len(candles))

        samples_out = []

        self.x = torch.randn(n, 10)
        self.y = (self.x.sum(dim=1) > 0).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
