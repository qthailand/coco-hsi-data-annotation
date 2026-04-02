import sys

from PyQt5.QtWidgets import QApplication

from .ui.window import PaintWindow


def run(argv=None):
    app = QApplication(argv or sys.argv)
    app.setStyle("Fusion")
    window = PaintWindow()
    window.show()
    return app.exec_()