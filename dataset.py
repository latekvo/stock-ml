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

    def __init__(
        self,
        history_count: int,
        prediction_count: int,
        # TODO: impl `training_ratio: float = 0.8`
        training_cutoff_date: int = 20230101,
    ):
        self.net_inputs = history_count * 5
        self.net_outputs = prediction_count * 2

        eval_pairs: list[CandleTrainingPair] = []
        training_pairs: list[CandleTrainingPair] = []
        history_offsets = fib_offsets(history_count)
        prediction_offsets = fib_offsets(prediction_count)
        anchor_start_index = history_offsets[-1]

        skipped_files: list[str] = []
        skipped_pairs: int = 0
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
                pair = CandleTrainingPair(
                    candle_sequence,
                    anchor_index,
                    history_offsets,
                    prediction_offsets,
                )

                if pair.first_prediction_date > training_cutoff_date:
                    eval_pairs.append(pair)
                elif pair.last_prediction_date < training_cutoff_date:
                    training_pairs.append(pair)
                else:
                    # eval-training overlap
                    # TODO: Figure out if we have to discard them or not
                    skipped_pairs += 1

        print("Skipped", len(skipped_files), "files:", *skipped_files)
        print("Skipped", skipped_pairs, "pairs: Eval/training overlap.")

        training_n = len(training_pairs)
        self.t_x = torch.zeros(training_n, self.net_inputs)
        self.t_y = torch.zeros(training_n, self.net_outputs)

        eval_n = len(eval_pairs)
        self.e_x = torch.zeros(eval_n, self.net_inputs)
        self.e_y = torch.zeros(eval_n, self.net_outputs)

        progress_bar = tqdm(range(training_n), desc="Merging training datasets")

        for i in progress_bar:
            self.t_x[i] = training_pairs[i].x
            self.t_y[i] = training_pairs[i].y

        progress_bar = tqdm(range(eval_n), desc="Merging eval datasets")

        for i in progress_bar:
            self.e_x[i] = eval_pairs[i].x
            self.e_y[i] = eval_pairs[i].y

    def __len__(self):
        return len(self.x)
