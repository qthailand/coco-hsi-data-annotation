from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)


def _jet_color(value):
    value = max(0.0, min(1.0, float(value)))
    red = max(0.0, min(1.0, 1.5 - abs(4.0 * value - 3.0)))
    green = max(0.0, min(1.0, 1.5 - abs(4.0 * value - 2.0)))
    blue = max(0.0, min(1.0, 1.5 - abs(4.0 * value - 1.0)))
    return QColor(int(round(red * 255)), int(round(green * 255)), int(round(blue * 255)))


class ClassTable(QWidget):
    class_changed = pyqtSignal(int, str, QColor)

    _DEFAULTS = [
        (1, "Class 1", _jet_color(0.12)),
        (2, "Class 2", _jet_color(0.88)),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QLabel("🏷️  Class Labels")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(
            "font-weight:bold; font-size:13px; padding:5px;"
            "background:#2b2b2b; color:#e0e0e0; border-radius:4px;"
        )
        layout.addWidget(header)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["ID", "Class Name", "Color"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self._table.setColumnWidth(0, 60)
        self._table.setColumnWidth(2, 80)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setStyleSheet(
            "QTableWidget { background:#1e1e1e; color:#e0e0e0; gridline-color:#333; }"
            "QHeaderView::section { background:#2b2b2b; color:#ccc; padding:4px; }"
            "QTableWidget::item:selected { background:#3a3a5c; }"
        )
        self._table.itemSelectionChanged.connect(self._on_selection)
        self._table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._table, stretch=1)

        buttons = QHBoxLayout()
        add_button = QPushButton("➕  Add")
        remove_button = QPushButton("🗑  Remove")
        add_button.clicked.connect(self._add_row)
        remove_button.clicked.connect(self._remove_row)
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        layout.addLayout(buttons)

        self._active_lbl = QLabel("—")
        self._active_lbl.setAlignment(Qt.AlignCenter)
        self._active_lbl.setStyleSheet(
            "font-size:12px; padding:4px; border-radius:3px; color:#aaa;"
        )
        layout.addWidget(self._active_lbl)

        self._suspend_item_change = False
        self._classes_map = {}

        for class_id, name, color in self._DEFAULTS:
            self._insert_row(class_id, name, color)
        self._table.selectRow(0)

    def _insert_row(self, class_id, name, color):
        row = self._table.rowCount()
        self._table.insertRow(row)

        id_item = QTableWidgetItem(str(class_id))
        id_item.setFlags(id_item.flags() | Qt.ItemIsEditable)
        self._table.setItem(row, 0, id_item)

        name_item = QTableWidgetItem(name)
        name_item.setFlags(name_item.flags() | Qt.ItemIsEditable)
        self._table.setItem(row, 1, name_item)

        button = QPushButton()
        button.setFixedHeight(26)
        button.setStyleSheet(
            "background:{0}; border:1px solid #555; border-radius:3px;".format(color.name())
        )
        button.setProperty("_color", color)
        button.clicked.connect(lambda _, btn=button: self._pick_color_for_button(btn))
        self._table.setCellWidget(row, 2, button)

        self._refresh_classes_map()

    def _row_of_button(self, button):
        for row in range(self._table.rowCount()):
            if self._table.cellWidget(row, 2) is button:
                return row
        return -1

    def _pick_color_for_button(self, button):
        row = self._row_of_button(button)
        if row >= 0:
            self._pick_color(row)

    def _pick_color(self, row):
        button = self._table.cellWidget(row, 2)
        if button is None:
            return
        old_color = button.property("_color") or QColor(Qt.red)
        color = QColorDialog.getColor(old_color, self, "เลือกสีคลาส")
        if color.isValid():
            if self._color_is_used_elsewhere(color, row):
                QMessageBox.warning(
                    self,
                    "สีซ้ำ",
                    "สีนี้ถูกใช้แล้วโดย class อื่น กรุณาเลือกสีที่ไม่ซ้ำ",
                )
                return
            button.setStyleSheet(
                "background:{0}; border:1px solid #555; border-radius:3px;".format(color.name())
            )
            button.setProperty("_color", color)
            self._refresh_classes_map()
            if row == self._table.currentRow():
                self._emit_active()

    def add_class_dialog(self):
        class_id = self._next_available_id()
        default_name = "Class {0}".format(self._table.rowCount() + 1)
        default_color = self._next_unique_jet_color()

        dialog = QDialog(self)
        dialog.setWindowTitle("เพิ่ม class")
        dlg_layout = QVBoxLayout(dialog)
        dlg_layout.setContentsMargins(12, 12, 12, 12)
        dlg_layout.setSpacing(10)

        form_layout = QVBoxLayout()

        name_row = QHBoxLayout()
        name_label = QLabel("ชื่อ class:", dialog)
        name_label.setFixedWidth(80)
        name_edit = QLineEdit(dialog)
        name_edit.setText(default_name)
        name_row.addWidget(name_label)
        name_row.addWidget(name_edit)

        color_row = QHBoxLayout()
        color_label = QLabel("สี class:", dialog)
        color_label.setFixedWidth(80)
        color_btn = QPushButton(dialog)
        color_btn.setFixedSize(80, 30)
        color_btn.setStyleSheet(
            "background:{0}; border:1px solid #555; border-radius:4px;".format(default_color.name())
        )
        color_btn.setProperty("_color", default_color)
        color_row.addWidget(color_label)
        color_row.addWidget(color_btn)

        form_layout.addLayout(name_row)
        form_layout.addLayout(color_row)

        def choose_color():
            selected = QColorDialog.getColor(default_color, dialog, "เลือกสี class")
            if selected.isValid():
                if self._color_is_used_elsewhere(selected, -1):
                    QMessageBox.warning(
                        self,
                        "สีซ้ำ",
                        "สีนี้ถูกใช้แล้ว กรุณาเลือกสีใหม่",
                    )
                    return
                color_btn.setStyleSheet(
                    "background:{0}; border:1px solid #555; border-radius:4px;".format(selected.name())
                )
                color_btn.setProperty("_color", selected)

        color_btn.clicked.connect(choose_color)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)

        dlg_layout.addLayout(form_layout)
        dlg_layout.addWidget(btn_box)

        if dialog.exec_() != QDialog.Accepted:
            return None

        name = name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "ชื่อว่าง", "กรุณาใส่ชื่อ class")
            return None

        color = color_btn.property("_color") or default_color
        self.add_class(class_id=class_id, name=name, color=color)
        return class_id

    def _add_row(self):
        self.add_class_dialog()

    def add_class(self, class_id=None, name=None, color=None):
        if class_id is None:
            class_id = self._next_available_id()
        if name is None:
            name = "Class {0}".format(self._table.rowCount() + 1)
        if color is None:
            color = self._next_unique_jet_color()
        self._insert_row(class_id, name, color)
        self._table.selectRow(self._table.rowCount() - 1)
        self._emit_active()

    def _remove_row(self):
        row = self._table.currentRow()
        if row >= 0:
            class_id = self._class_id_from_row(row)
            if class_id in self._classes_map:
                del self._classes_map[class_id]
            self._table.removeRow(row)
            row_count = self._table.rowCount()
            if row_count > 0:
                self._table.selectRow(min(row, row_count - 1))

    def _on_selection(self):
        self._emit_active()

    def _on_item_changed(self, item):
        if self._suspend_item_change:
            return
        if item.column() == 0:
            self._normalize_id_item(item)
        self._refresh_classes_map()
        if item.row() == self._table.currentRow() and item.column() in (0, 1):
            self._emit_active()

    def _next_available_id(self):
        used = set()
        for row in range(self._table.rowCount()):
            class_id = self._class_id_from_row(row)
            if class_id is not None:
                used.add(class_id)
        candidate = 1
        while candidate in used:
            candidate += 1
        return candidate

    def _next_unique_jet_color(self):
        used_colors = {
            self._color_key(self._row_color(row)) for row in range(self._table.rowCount())
        }
        sample_count = max(8, self._table.rowCount() + 1)
        for offset in range(sample_count):
            sample_index = (offset + self._table.rowCount()) % sample_count
            if sample_count > 1:
                fraction = sample_index / float(sample_count - 1)
            else:
                fraction = 0.0
            color = _jet_color(fraction)
            if self._color_key(color) not in used_colors:
                return color

        return _jet_color((self._table.rowCount() % 256) / 255.0)

    def _row_color(self, row):
        button = self._table.cellWidget(row, 2)
        if button is None:
            return QColor(Qt.red)
        color = button.property("_color")
        return QColor(color) if isinstance(color, QColor) else QColor(Qt.red)

    def _color_key(self, color):
        return (color.red(), color.green(), color.blue(), color.alpha())

    def _color_is_used_elsewhere(self, color, row_to_ignore):
        key = self._color_key(color)
        for row in range(self._table.rowCount()):
            if row == row_to_ignore:
                continue
            if self._color_key(self._row_color(row)) == key:
                return True
        return False

    def _normalize_id_item(self, item):
        text = (item.text() or "").strip()
        if text.isdigit() and int(text) > 0:
            normalized = str(int(text))
        else:
            normalized = str(item.row() + 1)
        if normalized != item.text():
            self._suspend_item_change = True
            item.setText(normalized)
            self._suspend_item_change = False

    def _refresh_classes_map(self):
        self._classes_map = {}
        for row in range(self._table.rowCount()):
            class_id = self._class_id_from_row(row)
            if class_id is None:
                continue
            name_item = self._table.item(row, 1)
            name = name_item.text() if name_item else f"Class {row + 1}"
            color = self._row_color(row)
            self._classes_map[class_id] = {
                "id": class_id,
                "name": name,
                "color": color,
            }

    def get_classes_dict(self):
        return dict(self._classes_map)

    def _class_id_from_row(self, row):
        item = self._table.item(row, 0)
        if item is None:
            return None
        text = (item.text() or "").strip()
        if not text.isdigit():
            return None
        return int(text)

    def validate_class_ids(self):
        if self._table.rowCount() == 0:
            return False, "ยังไม่มี class ให้บันทึก"

        seen = set()
        for row in range(self._table.rowCount()):
            class_id = self._class_id_from_row(row)
            if class_id is None:
                return False, "ID ของ class แถว {0} ต้องเป็นจำนวนเต็ม".format(row + 1)
            if class_id <= 0:
                return False, "ID ของ class แถว {0} ต้องมากกว่า 0 (0 สงวนไว้เป็น background)".format(
                    row + 1
                )
            if class_id > 255:
                return False, "ID ของ class แถว {0} ต้องไม่เกิน 255".format(row + 1)
            if class_id in seen:
                return False, "ID ซ้ำกันที่ค่า {0}".format(class_id)
            seen.add(class_id)
        return True, ""

    def _emit_active(self):
        row = self._table.currentRow()
        if row < 0:
            return
        class_id = self._class_id_from_row(row)
        if class_id is None:
            class_id = row + 1
        name_item = self._table.item(row, 1)
        name = name_item.text() if name_item else "Class {0}".format(row + 1)
        button = self._table.cellWidget(row, 2)
        color = button.property("_color") if button else QColor(Qt.red)
        fg = "#000" if color.lightness() > 128 else "#fff"
        self._active_lbl.setText("Active: <b>{0}</b> (ID={1})".format(name, class_id))
        self._active_lbl.setStyleSheet(
            "font-size:12px; font-weight:bold; padding:4px; border-radius:3px;"
            "background:{0}; color:{1};".format(color.name(), fg)
        )
        self.class_changed.emit(class_id, name, color)

    def active_color(self):
        row = self._table.currentRow()
        if row < 0:
            return QColor(Qt.red)
        button = self._table.cellWidget(row, 2)
        return button.property("_color") if button else QColor(Qt.red)

    def active_name(self):
        row = self._table.currentRow()
        if row < 0:
            return "Unknown"
        item = self._table.item(row, 1)
        return item.text() if item else "Class {0}".format(row + 1)

    def active_class_id(self):
        row = self._table.currentRow()
        if row < 0:
            return 1
        class_id = self._class_id_from_row(row)
        return class_id if class_id is not None else row + 1

    def get_all(self):
        result = []
        for row in range(self._table.rowCount()):
            class_id = self._class_id_from_row(row)
            if class_id is None:
                class_id = row + 1
            item = self._table.item(row, 1)
            name = item.text() if item else "Class {0}".format(row + 1)
            button = self._table.cellWidget(row, 2)
            color = button.property("_color") if button else QColor(Qt.red)
            result.append((class_id, name, color))
        return result

    def get_classes_dict(self):
        return dict(self._classes_map)
