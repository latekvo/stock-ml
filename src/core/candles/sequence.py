from .candle import Candle


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

    def __len__(self):
        return len(self.candles)

    def get_candle(self, idx):
        return self.get_merged_candle(idx, idx)

    def get_merged_candle(self, from_idx: int, to_idx_incl: int):
        assert from_idx >= 0
        assert to_idx_incl < len(self.candles)
        assert from_idx <= to_idx_incl

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
