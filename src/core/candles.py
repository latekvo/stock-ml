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
    ticker: str
    interval: str
    date_start: int = None
    date_end: int = None
    candles: list[Candle] = []

    def __init__(self, ticker, interval, candles: list[Candle]):
        self.ticker = ticker
        self.interval = interval

        for candle in candles:
            if self.date_start is None or candle.date < self.date_start:
                self.date_start = candle.date
            if self.date_end is None or candle.date > self.date_end:
                self.date_end = candle.date
            self.candles.append(candle)

        assert self.date_start is not None and self.date_end is not None


class CandleDataset:
    anchor_date: int
    candle_history: list[Candle]
    candle_prediction: list[Candle]

    def __init__(self, sequence, anchor_date, history_offsets, prediction_offsets):
        pass
