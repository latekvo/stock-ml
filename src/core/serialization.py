from src.core.candles import Candle, CandleSequence


def serialize_stooq_file(filepath: str) -> CandleSequence:
    with open(filepath, "r") as f:
        # 1st line is header
        contents = f.read().split("\n")[1:-1]

        ticker = None
        interval = None
        candles = []

        for datapoint in contents:
            if len(datapoint) < 2:
                break  # EOF

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

            candles.append(
                Candle(
                    int(date),
                    int(time),
                    float(price_open),
                    float(price_close),
                    float(price_high),
                    float(price_low),
                    float(volume),
                )
            )

        return CandleSequence(ticker, interval, candles)
