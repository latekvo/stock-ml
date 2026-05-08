from src.core.telemetry import TelemetryRegistry
import pyqtgraph as pg


def run_visualisation_app(
    telemetry_registry: TelemetryRegistry,
):
    curves = {}
    names = telemetry_registry.get_loss_entries()

    def refresh():
        for name in names:
            curves[name].setData(telemetry_registry.get_loss_history(name))

    pg.mkQApp()
    win = pg.GraphicsLayoutWidget(show=True)
    loss_plot = win.addPlot()
    loss_plot.addLegend()

    for name in names:
        color = telemetry_registry.get_loss_color(name)
        curves[name] = loss_plot.plot(pen=color, name=name)

    timer = pg.QtCore.QTimer(curves[names[0]])
    timer.timeout.connect(refresh)
    timer.start(20)

    pg.exec()
