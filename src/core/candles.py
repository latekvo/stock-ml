from typing import TypedDict


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

        for candle in candle_list:
            candle_date = candle["date_start"]

            # TODO: We aren't handling candle merged count because we don't need to for now

            if self.date_start is None or candle_date < self.date_start:
                self.date_start = candle_date
            if self.date_end is None or candle_date > self.date_end:
                self.date_end = candle_date
            self.candles.append(candle)

        assert self.date_start is not None and self.date_end is not None


class CandleDataset:
    anchor_date: int
    candle_history: list[Candle]
    candle_prediction: list[Candle]

    def __init__(self, sequence, anchor_date, history_offsets, prediction_offsets):
        pass
