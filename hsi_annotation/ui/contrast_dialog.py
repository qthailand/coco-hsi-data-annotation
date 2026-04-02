from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSlider,
    QVBoxLayout,
)


class ContrastDialog(QDialog):
    preview_changed = pyqtSignal(float, float)

    def __init__(self, low_cut, high_cut, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RGB Contrast")
        self.setModal(True)
        self.resize(360, 200)
        self._scale = 10

        layout = QVBoxLayout(self)

        hint = QLabel(
            "ปรับ percentile cut ของภาพ RGB เพื่อดึงรายละเอียดจาก cube ที่มืดหรือคอนทราสต่ำ"
        )
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(hint)

        form = QFormLayout()
        self._low_cut = self._create_slider(0, 990, low_cut)
        self._low_value = QLabel()
        form.addRow("Low cut", self._build_slider_row(self._low_cut, self._low_value))

        self._high_cut = self._create_slider(10, 1000, high_cut)
        self._high_value = QLabel()
        form.addRow("High cut", self._build_slider_row(self._high_cut, self._high_value))
        layout.addLayout(form)

        self._low_cut.valueChanged.connect(self._on_low_changed)
        self._high_cut.valueChanged.connect(self._on_high_changed)
        self._sync_labels()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self):
        return self._low_cut.value() / self._scale, self._high_cut.value() / self._scale

    def _create_slider(self, minimum, maximum, value):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setSingleStep(1)
        slider.setPageStep(5)
        slider.setValue(int(round(float(value) * self._scale)))
        return slider

    def _build_slider_row(self, slider, value_label):
        row = QHBoxLayout()
        row.addWidget(slider, stretch=1)
        value_label.setMinimumWidth(52)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(value_label)
        return row

    def _sync_labels(self):
        low_cut, high_cut = self.values()
        self._low_value.setText("{0:.1f} %".format(low_cut))
        self._high_value.setText("{0:.1f} %".format(high_cut))

    def _emit_preview(self):
        self._sync_labels()
        low_cut, high_cut = self.values()
        self.preview_changed.emit(low_cut, high_cut)

    def _on_low_changed(self, value):
        if value >= self._high_cut.value():
            self._high_cut.blockSignals(True)
            self._high_cut.setValue(min(value + 1, self._high_cut.maximum()))
            self._high_cut.blockSignals(False)
        self._emit_preview()

    def _on_high_changed(self, value):
        if value <= self._low_cut.value():
            self._low_cut.blockSignals(True)
            self._low_cut.setValue(max(value - 1, self._low_cut.minimum()))
            self._low_cut.blockSignals(False)
        self._emit_preview()

    def _accept_if_valid(self):
        low_cut, high_cut = self.values()
        if low_cut >= high_cut:
            QMessageBox.warning(self, "ค่าไม่ถูกต้อง", "Low cut ต้องน้อยกว่า High cut")
            return
        self.accept()