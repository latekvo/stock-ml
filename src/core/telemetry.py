import numpy as np

LOSS_HISTORY_WINDOW = 1000


class TelemetryRegistry:
    _loss_history: np.ndarray[tuple[int], np.dtype[np.float64]]

    def __init__(self):
        self._loss_history = np.zeros(LOSS_HISTORY_WINDOW)

    @property
    def loss_history(self):
        return self._loss_history

    def log_loss(self, loss: np.float64):
        # TODO: Expand instead of sliding window
        self._loss_history[:-1] = self._loss_history[1:]
        self._loss_history[-1] = loss
