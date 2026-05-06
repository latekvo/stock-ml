from typing import TypedDict

import torch

from src.core.utils import offsets_to_ranges


class Candle(TypedDict):
    date_start: int
    merged_count: int  # =1 default; >1 for merged
    price_open: float
    price_close: float
    price_high: float
    price_low: float
    volume: float


class CandleSequence:
    def __init__(self, ticker, interval, candle_list: list[Candle]):
        self.ticker = ticker
        self.interval = interval
        self.date_start: int | None = None
        self.date_end: int | None = None
        self.candles: list[Candle] = []

        assert len(candle_list) > 0

        for candle in candle_list:
            candle_date = candle["date_start"]

            # TODO: We aren't handling candle merged count because we don't need to for now

            if self.date_start is None or candle_date < self.date_start:
                self.date_start = candle_date
            if self.date_end is None or candle_date > self.date_end:
                self.date_end = candle_date
            self.candles.append(candle)

        assert self.date_start is not None and self.date_end is not None

    def get_merged_candle(self, from_idx: int, to_idx_incl: int):
        first = self.candles[from_idx]
        last = self.candles[to_idx_incl]

        merged = Candle(
            merged_count=1,
            date_start=first["date_start"],
            price_open=first["price_close"],
            price_close=last["price_close"],
            price_high=last["price_high"],
            price_low=last["price_low"],
            volume=last["volume"],
        )

        for i in range(from_idx, to_idx_incl):
            candle = self.candles[i]
            merged["merged_count"] += 1
            merged["price_high"] = max(candle["price_high"], merged["price_high"])
            merged["price_low"] = min(candle["price_low"], merged["price_low"])
            merged["volume"] += candle["volume"]

        return merged


class CandleTrainingPair:
    def __init__(
        self,
        candle_sequence: CandleSequence,
        anchor_index: int,
        history_offsets: list[int],
        prediction_offsets: list[int],
    ):
        self.anchor_index = anchor_index
        self.candle_history: list[Candle] = []
        self.candle_prediction: list[Candle] = []

        hist_ranges = offsets_to_ranges(history_offsets)
        pred_ranges = offsets_to_ranges(prediction_offsets)
        a = anchor_index

        for r in hist_ranges:
            candle = candle_sequence.get_merged_candle(a - r[1], a - r[0])
            self.candle_history.append(candle)

        for r in pred_ranges:
            candle = candle_sequence.get_merged_candle(a + r[0], a + r[1])
            self.candle_prediction.append(candle)

    @property
    def x(self) -> torch.Tensor:
        return torch.Tensor(self.candle_history)

    @property
    def y(self) -> torch.Tensor:
        return torch.Tensor(self.candle_prediction)
