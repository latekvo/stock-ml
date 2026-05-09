import numpy as np

LOSS_HISTORY_WINDOW = 1000

LossHistory = np.ndarray[tuple[int], np.dtype[np.float64]]
Color = str | tuple[int, int, int]


class _LossHistoryData:
    loss_history: LossHistory
    color: Color

    def __init__(self, color: Color):
        self.loss_history = np.zeros(LOSS_HISTORY_WINDOW)
        self.color = color


class TelemetryRegistry:
    _loss_registry: dict[str, _LossHistoryData]

    def __init__(self):
        self._loss_history = np.zeros(LOSS_HISTORY_WINDOW)
        self._eval_loss_history = np.zeros(LOSS_HISTORY_WINDOW)
        self._loss_registry = {}

    def get_loss_history(self, entry_name: str) -> LossHistory:
        assert entry_name in self._loss_registry
        return self._loss_registry[entry_name].loss_history

    def get_loss_color(self, entry_name: str) -> Color:
        assert entry_name in self._loss_registry
        return self._loss_registry[entry_name].color

    def get_loss_entries(self) -> list[str]:
        return list(self._loss_registry.keys())

    def add_loss_entry(self, entry_name: str, color: Color = "w") -> None:
        self._loss_registry[entry_name] = _LossHistoryData(color)

    def log_loss(self, entry_name: str, loss: np.float64):
        assert entry_name in self._loss_registry
        data = self._loss_registry[entry_name]
        data.loss_history[:-1] = data.loss_history[1:]
        data.loss_history[-1] = loss
