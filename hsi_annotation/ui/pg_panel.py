import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)


class PgPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._first = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QLabel("📊  PyQtGraph – GT Mask & Spectrum")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(
            "font-weight:bold; font-size:13px; padding:5px;"
            "background:#2b2b2b; color:#e0e0e0; border-radius:4px;"
        )
        layout.addWidget(header)

        self._img_view = pg.ImageView()
        self._img_view.ui.roiBtn.hide()
        self._img_view.ui.menuBtn.hide()
        self._img_view.ui.roiPlot.hide()
        layout.addWidget(self._img_view, stretch=3)

        spectrum_label = QLabel("📈  Spectrum at Cursor")
        spectrum_label.setAlignment(Qt.AlignCenter)
        spectrum_label.setStyleSheet(
            "font-weight:bold; color:#ccc; background:#2b2b2b;"
            "padding:3px; border-radius:3px;"
        )
        layout.addWidget(spectrum_label)

        self._spec_plot = pg.PlotWidget()
        self._spec_plot.setBackground("#1e1e1e")
        self._spec_plot.setLabel("left", "Reflectance")
        self._spec_plot.setLabel("bottom", "Band index")
        self._spec_plot.showGrid(x=True, y=True, alpha=0.25)
        self._spec_plot.addLegend(offset=(10, 10))
        self._spec_curve = self._spec_plot.plot(
            [], [], pen=pg.mkPen((255, 255, 255), width=1.5), name="At cursor"
        )
        self._class_curves = {}
        layout.addWidget(self._spec_plot, stretch=2)

        self._spec_lbl = QLabel("คลิกบน canvas เพื่อดู spectrum")
        self._spec_lbl.setAlignment(Qt.AlignCenter)
        self._spec_lbl.setStyleSheet("font-size:10px; color:#888; padding:2px;")
        layout.addWidget(self._spec_lbl)

        self._progress_lbl = QLabel("คลิกบน canvas เพื่อดู spectrum")
        self._progress_lbl.setAlignment(Qt.AlignCenter)
        self._progress_lbl.setStyleSheet("font-size:10px; color:#a0a0a0; padding:2px;")
        layout.addWidget(self._progress_lbl)

    def update_from_mask(self, mask):
        arr = self._to_np(mask)
        self._img_view.setImage(
            arr[:, :, :3],
            autoRange=self._first,
            autoLevels=self._first,
            autoHistogramRange=self._first,
        )
        self._first = False

    def update_spectrum(self, x, y, arr):
        self._spec_curve.setData(np.arange(len(arr), dtype=float), arr.astype(float))
        self._spec_lbl.setText(
            "cursor: col={0}  row={1}  bands={2}  mean={3:.4f}".format(
                x, y, len(arr), arr.mean()
            )
        )

    def set_spectrum_status(self, text):
        self._progress_lbl.setText(text)

    def reset_spectrum_status(self):
        self._progress_lbl.setText("คลิกบน canvas เพื่อดู spectrum")

    def update_class_spectra(self, class_data):
        self.reset_spectrum_status()
        for curve in self._class_curves.values():
            self._spec_plot.removeItem(curve)
        self._class_curves.clear()

        for name, color, avg in class_data:
            if avg is None:
                continue
            curve = self._spec_plot.plot(
                np.arange(len(avg), dtype=float),
                avg.astype(float),
                pen=pg.mkPen((color.red(), color.green(), color.blue()), width=2.2),
                name=name,
            )
            self._class_curves[name] = curve

        parts = [
            "<span style='color:#{0:02x}{1:02x}{2:02x}'>{3}</span>".format(
                color.red(), color.green(), color.blue(), name
            )
            for name, color, avg in class_data
            if avg is not None
        ]
        self._spec_plot.setTitle("  ".join(parts) if parts else "")

    @staticmethod
    def _to_np(img):
        rgba = img.convertToFormat(QImage.Format_RGBA8888)
        width, height = rgba.width(), rgba.height()
        ptr = rgba.bits()
        ptr.setsize(height * width * 4)
        return np.frombuffer(ptr, np.uint8).reshape((height, width, 4)).copy()