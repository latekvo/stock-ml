import torch

from .sequence import CandleSequence
from .candle import Candle
from src.core.utils import offsets_to_ranges


def normalize_candle(c: Candle, anchor: Candle):
    c["volume"] = c["volume"] / anchor["volume"]
    c["price_open"] = c["price_open"] / anchor["price_open"]
    c["price_close"] = c["price_close"] / anchor["price_close"]
    c["price_low"] = c["price_low"] / anchor["price_low"]
    c["price_high"] = c["price_high"] / anchor["price_high"]


def prepare_anchor(anchor_index: int, sequence: CandleSequence) -> Candle:
    anchor = sequence.get_candle(anchor_index)
    anchor["volume"] = max(1, anchor["volume"])
    anchor["price_open"] = max(0.01, anchor["price_open"])
    anchor["price_close"] = max(0.01, anchor["price_close"])
    anchor["price_low"] = max(0.01, anchor["price_low"])
    anchor["price_high"] = max(0.01, anchor["price_high"])
    return anchor


def prepare_input_sequence(sequence: CandleSequence) -> CandleSequence:
    pass


def prepare_output_sequence(sequence: CandleSequence) -> CandleSequence:
    pass


def flatten_sequence(sequence: CandleSequence):
    pass


class CandleTrainingPair:
    def __init__(
        self,
        candle_sequence: CandleSequence,
        anchor_index: int,
        history_offsets: list[int],
        prediction_offsets: list[int],
    ):
        self.anchor_index = anchor_index
        self.flattened_history: list[float] = []
        self.flattened_prediction: list[float] = []

        candle_history: list[Candle] = []
        candle_prediction: list[Candle] = []

        hist_ranges = offsets_to_ranges(history_offsets)
        pred_ranges = offsets_to_ranges(prediction_offsets)
        a = anchor_index

        # FIXME: NORMALIZE EVERY PRICE TO [-1;1]
        #  This is to ignore absolute movement velocities
        #   to focus on relative movement velocities

        anchor = prepare_anchor(a, candle_sequence)

        for r in hist_ranges:
            candle = candle_sequence.get_merged_candle(a - r[1], a - r[0])
            normalize_candle(candle, anchor)
            candle_history.append(candle)

        for r in pred_ranges:
            candle = candle_sequence.get_merged_candle(a + r[0], a + r[1])
            normalize_candle(candle, anchor)
            candle_prediction.append(candle)

        self.first_prediction_date = candle_prediction[0]["date_start"]
        self.last_prediction_date = candle_prediction[-1]["date_start"] + 1

        for candle in candle_history:
            self.flattened_history.append(candle["volume"])
            self.flattened_history.append(candle["price_open"])
            self.flattened_history.append(candle["price_close"])
            self.flattened_history.append(candle["price_low"])
            self.flattened_history.append(candle["price_high"])

        for candle in candle_prediction:
            self.flattened_prediction.append(candle["price_low"])
            self.flattened_prediction.append(candle["price_high"])

    @property
    def x(self) -> torch.Tensor:
        return torch.Tensor(self.flattened_history)

    @property
    def y(self) -> torch.Tensor:
        return torch.Tensor(self.flattened_prediction)
