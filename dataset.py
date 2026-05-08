import os
from pathlib import Path

from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from src.core.utils import fib_offsets
from src.core.serialization import serialize_stooq_file
from src.core.candles import CandleTrainingPair

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

        skipped_files: list[str] = []
        progress_bar = tqdm(range(len(files)), desc="Loading files")

        for file, _ in zip(files, progress_bar):
            filename = file.absolute().as_posix()

            try:
                candle_sequence = serialize_stooq_file(filename)
            except:
                skipped_files.append(str(file))
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

        print("Skipped", len(skipped_files), "files:", *skipped_files)

        n = len(training_pairs)
        self.x = torch.zeros(n, self.net_inputs)
        self.y = torch.zeros(n, self.net_outputs)

        progress_bar = tqdm(range(n), desc="Merging dataset")

        for i in progress_bar:
            self.x[i] = training_pairs[i].x
            self.y[i] = training_pairs[i].y

    def __len__(self):
        return len(self.x)
