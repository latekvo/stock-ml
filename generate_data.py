import os
from dataclasses import dataclass
from pathlib import Path

DATE_CUTOFF_FOR_EVAL = 20240101  # date format used by nc dataset

DATASET_PATH = ("daily", "pl", "nc stocks")

data_raw = os.path.join("data", "raw", *DATASET_PATH)
data_out = os.path.join("data", "training")
data_eval = os.path.join("data", "training")

files = list(Path(data_raw).iterdir())


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


def serialize_stooq_file(filepath: str):
    with open(filepath, "r") as f:
        # 1st line is header
        contents = f.read().split("\n")[1:-1]

        candles = []

        for datapoint in contents:
            (
                ticker,
                interval,
                date,
                time,
                price_open,
                price_high,
                price_low,
                price_close,
                volume,
                open_interval,
            ) = datapoint.split(",")

            candles.append(
                Candle(
                    ticker,
                    interval,
                    float(date),
                    float(time),
                    float(price_open),
                    float(price_close),
                    float(price_high),
                    float(price_low),
                    float(volume),
                    float(open_interval),
                )
            )

        return candles


for file in files:
    filename = file.absolute().as_posix()
    candles = serialize_stooq_file(filename)
    print(file.absolute().as_posix(), len(candles))

samples_out = []
