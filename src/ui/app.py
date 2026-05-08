from src.core.telemetry import TelemetryRegistry
import pyqtgraph as pg


def run_visualisation_app(
    telemetry_registry: TelemetryRegistry,
):
    def refresh():
        training_loss_curve.setData(telemetry_registry.loss_history)
        eval_loss_curve.setData(telemetry_registry.eval_loss_history)

    pg.mkQApp()

    win = pg.GraphicsLayoutWidget(show=True)

    loss_plot = win.addPlot()
    
    training_loss_curve = loss_plot.plot(pen='r')
    eval_loss_curve = loss_plot.plot(pen='w')
    
    timer = pg.QtCore.QTimer(training_loss_curve)
    timer.timeout.connect(refresh)
    timer.start(20)

    pg.exec()
