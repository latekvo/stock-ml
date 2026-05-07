from typing import TypedDict


class Candle(TypedDict):
    date_start: int
    merged_count: int  # =1 default; >1 for merged
    price_open: float
    price_close: float
    price_high: float
    price_low: float
    volume: float
