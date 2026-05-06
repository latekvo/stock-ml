import os
from pathlib import Path

from torch.utils.data import Dataset
import torch

from src.core.utils import fib_offsets
from src.core.serialization import serialize_stooq_file
from src.core.candles import CandleTrainingPair


class DummyDataset(Dataset):
    def __init__(self, n=1000):
        self.x = torch.randn(n, 10)
        self.y = (self.x.sum(dim=1) > 0).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


DATA_RAW = os.path.join("data", "daily", "pl", "nc stocks")
TRAINING_CUTOFF_DATE = 20220101
files = list(Path(DATA_RAW).iterdir())


class FinancialDataset(Dataset):
    def __init__(self, history_count: int, prediction_count: int):
        self.net_inputs = history_count * 5
        self.net_outputs = prediction_count * 2

        training_pairs: list[CandleTrainingPair] = []
        history_offsets = fib_offsets(history_count)
        prediction_offsets = fib_offsets(prediction_count)
        anchor_start_index = history_offsets[-1]

        for file in files:
            filename = file.absolute().as_posix()

            try:
                candle_sequence = serialize_stooq_file(filename)
            except:
                print("Skipping - Error when reading:", filename)
                continue

            anchor_end_index = len(candle_sequence) - prediction_offsets[-1]

            for anchor_index in range(anchor_start_index, anchor_end_index):
                candle_training_pair = CandleTrainingPair(
                    candle_sequence,
                    anchor_index,
                    history_offsets,
                    prediction_offsets,
                )

                training_pairs.append(candle_training_pair)

        n = len(training_pairs)
        self.x = torch.zeros(n, self.net_inputs)
        self.y = torch.zeros(n, self.net_outputs)

        for i in range(n):
            self.x[i] = training_pairs[i].x
            self.y[i] = training_pairs[i].y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
