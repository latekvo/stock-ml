from model import Net, AbstractNet
from src.core.candles import CandleSequence


def min_max_strategy(model: AbstractNet, candles: CandleSequence):
    # buy only if current candle < all future predictions
    # sell only if current candle > all future predictions
    predictions = model(candles)

    for candle in predictions:
        pass


def run_market_eval(model: Net, candles: CandleSequence):
    pass
