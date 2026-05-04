from dataclasses import dataclass


@dataclass
class Candle:
    ticker: str
    interval: str
    date: float
    time: float
    price_open: float
    price_close: float
    price_high: float
    price_low: float
    volume: float
    open_interval: float
