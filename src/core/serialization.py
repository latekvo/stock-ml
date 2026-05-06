from src.core.candles import Candle, CandleSequence


def serialize_stooq_file(filepath: str) -> CandleSequence:
    with open(filepath, "r") as f:
        # 1st line is header
        contents = f.read().split("\n")[1:-1]

        ticker: str | None = None
        interval = None
        candles: list[Candle] = []

        for datapoint in contents:
            (
                extracted_ticker,
                extracted_interval,
                date,
                time,
                price_open,
                price_high,
                price_low,
                price_close,
                volume,
                open_interval,
            ) = datapoint.split(",")

            ticker = extracted_ticker
            interval = extracted_interval

            candle = Candle(
                date_start=int(date),
                merged_count=int(time),
                price_open=float(price_open),
                price_close=float(price_close),
                price_high=float(price_high),
                price_low=float(price_low),
                volume=float(volume),
            )

            candles.append(candle)

        return CandleSequence(ticker, interval, candles)
