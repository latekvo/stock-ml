from src.core.telemetry import TelemetryRegistry
import pyqtgraph as pg


def run_visualisation_app(
    telemetry_registry: TelemetryRegistry,
):
    def refresh():
        curve.setData(telemetry_registry.loss_history)

    app = pg.mkQApp()

    win = pg.GraphicsLayoutWidget(show=True)

    plot = win.addPlot()
    curve = plot.plot()
    timer = pg.QtCore.QTimer(curve)
    timer.timeout.connect(refresh)
    timer.start(50)

    pg.exec()
