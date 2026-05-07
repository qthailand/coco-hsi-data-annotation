"""
ui/label_panel.py
-----------------
Left-side panel: QTableWidget synced two-way with LabelRegistry.

Columns:  ID (read-only) | Label (editable) | Color (click→picker) | 👁 (visible toggle)

Signals emitted:
    active_label_changed(label_id: int)
"""

import logging

from PyQt5.QtCore import Qt, pyqtSignal, QSize

log = logging.getLogger(__name__)
from PyQt5.QtGui import QColor, QFont, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QColorDialog, QLabel,
    QAbstractItemView, QSizePolicy, QFrame,
    QCheckBox,
)

from ..registry import LabelRegistry, AnnotationRegistry

# Column indices
COL_ID      = 0
COL_LABEL   = 1
COL_COLOR   = 2
COL_VISIBLE = 3
NUM_COLS    = 4


def _make_color_icon(color: QColor, w=40, h=18) -> QIcon:
    px = QPixmap(w, h)
    px.fill(color)
    return QIcon(px)


class _ColorButton(QPushButton):
    """Small color-swatch button; opens QColorDialog on click."""

    color_changed = pyqtSignal(QColor)

    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(QSize(48, 22))
        self.setFlat(False)
        self.setCursor(Qt.PointingHandCursor)
        self._refresh()
        self.clicked.connect(self._pick)

    def get_color(self) -> QColor:
        return QColor(self._color)

    def set_color(self, color: QColor):
        self._color = QColor(color)
        self._refresh()

    def _refresh(self):
        self.setIcon(_make_color_icon(self._color, 36, 14))
        self.setIconSize(QSize(36, 14))
        self.setStyleSheet(
            "QPushButton{{"
            "  border: 2px solid #666; border-radius: 3px;"
            "  background: {name};"
            "}}"
            "QPushButton:hover {{ border: 2px solid #ccc; }}"
            .format(name=self._color.name())
        )

    def _pick(self):
        c = QColorDialog.getColor(
            self._color, self, "เลือกสี label",
            QColorDialog.ShowAlphaChannel,
        )
        if c.isValid():
            c.setAlpha(255)          # force opaque for registry storage
            self._color = c
            self._refresh()
            self.color_changed.emit(c)


class LabelPanel(QWidget):
    """
    Left-side label manager.

    Signals
    -------
    active_label_changed(label_id: int)
    """

    active_label_changed = pyqtSignal(int)

    def __init__(self, label_registry: LabelRegistry,
                 annotation_registry: AnnotationRegistry,
                 parent=None):
        super().__init__(parent)
        self._labels  = label_registry
        self._annots  = annotation_registry
        self._active_id = None
        self._updating  = False   # guard recursive signal loops

        self._build_ui()
        self._connect_registry()
        self._rebuild_table()     # populate from existing registry state

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.setMinimumWidth(230)
        self.setMaximumWidth(340)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 8, 6, 6)
        root.setSpacing(5)

        # Table
        self._table = QTableWidget(0, NUM_COLS)
        self._table.setHorizontalHeaderLabels(["ID", "Label", "Color", "👁"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(COL_ID,      QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(COL_LABEL,   QHeaderView.Stretch)
        hh.setSectionResizeMode(COL_COLOR,   QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(COL_VISIBLE, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        # Only label column is editable (ID, Color, Visible handled by widgets)
        self._table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked
        )
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setStyleSheet(
            "QTableWidget {"
            "  border: 1px solid #444;"
            "  border-radius: 4px;"
            "  background: #1e1e1e;"
            "  color: #ddd;"
            "  alternate-background-color: #252525;"
            "}"
            "QTableWidget::item { padding: 2px 4px; }"
            "QTableWidget::item:selected {"
            "  background: #2d6ea4; color: white;"
            "}"
            "QHeaderView::section {"
            "  background: #2b2b2b; color: #bbb; padding: 4px;"
            "  border: none; border-bottom: 1px solid #444;"
            "}"
        )
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemChanged.connect(self._on_item_changed)
        root.addWidget(self._table)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self._btn_add = QPushButton("＋ Add")
        self._btn_add.setToolTip("Add label  [A]")
        self._btn_add.setShortcut("A")
        self._btn_add.clicked.connect(self.add_label)
        self._btn_del = QPushButton("－ Remove")
        self._btn_del.setToolTip("Remove selected label  [Del]")
        self._btn_del.clicked.connect(self.remove_selected)
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_del)
        root.addLayout(btn_row)

        # Active label indicator bar
        self._active_bar = QLabel("Active: —")
        self._active_bar.setAlignment(Qt.AlignCenter)
        self._active_bar.setWordWrap(True)
        self._active_bar.setStyleSheet(
            "QLabel {"
            "  background: #2d2d2d; border-radius: 4px;"
            "  padding: 5px; color: #ccc; font-size: 11px;"
            "}"
        )
        root.addWidget(self._active_bar)

    # ------------------------------------------------------------------
    # Registry → table  (one-way: registry changes update the table)
    # ------------------------------------------------------------------

    def _connect_registry(self):
        self._labels.label_added.connect(self._on_label_added)
        self._labels.label_removed.connect(self._on_label_removed)
        self._labels.label_changed.connect(self._on_label_changed_registry)
        self._labels.reset.connect(self._rebuild_table)

    def _rebuild_table(self):
        self._updating = True
        self._table.clearContents()
        self._table.setRowCount(0)
        for lid in self._labels.ids():
            self._append_row(lid)
        self._updating = False
        # restore selection
        if self._active_id and self._active_id in self._labels:
            self._select_row_for(self._active_id)
        elif self._labels.ids():
            self._select_row_for(self._labels.ids()[0])
        self._log_state("rebuild_table")

    def _on_label_added(self, label_id):
        self._updating = True
        self._append_row(label_id)
        self._updating = False
        self._select_row_for(label_id)
        self._log_state("label_added id={}".format(label_id))

    def _on_label_removed(self, label_id):
        row = self._row_for(label_id)
        if row < 0:
            return
        self._updating = True
        self._table.removeRow(row)
        self._updating = False
        # auto-select nearest remaining
        n = self._table.rowCount()
        if n > 0:
            new_row = min(row, n - 1)
            self._table.selectRow(new_row)
        else:
            self._active_id = None
            self._active_bar.setText("Active: —")
            self._active_bar.setStyleSheet(
                "QLabel { background:#2d2d2d; border-radius:4px; padding:5px; color:#ccc; font-size:11px; }"
            )
        self._log_state("label_removed id={}".format(label_id))

    def _on_label_changed_registry(self, label_id):
        """Registry changed externally → refresh that row's widgets."""
        row = self._row_for(label_id)
        if row < 0:
            return
        self._updating = True
        # update name item
        name_item = self._table.item(row, COL_LABEL)
        if name_item:
            name_item.setText(self._labels.name(label_id))
        # update color button
        btn = self._table.cellWidget(row, COL_COLOR)
        if btn:
            btn.set_color(self._labels.qcolor(label_id, alpha=255))
        self._updating = False
        if label_id == self._active_id:
            self._update_active_bar(label_id)
        self._log_state("label_changed id={}".format(label_id))

    # ------------------------------------------------------------------
    # Table → registry  (two-way: user edits update registry)
    # ------------------------------------------------------------------

    def _on_selection_changed(self):
        if self._updating:
            return
        row = self._table.currentRow()
        if row < 0:
            return
        lid = self._label_id_at_row(row)
        if lid is None:
            return
        if lid != self._active_id:
            self._active_id = lid
            self._update_active_bar(lid)
            self.active_label_changed.emit(lid)

    def _on_item_changed(self, item):
        if self._updating:
            return
        if item.column() != COL_LABEL:
            return
        lid = self._label_id_at_row(item.row())
        if lid is None:
            return
        new_name = item.text().strip()
        if not new_name:
            # revert
            self._updating = True
            item.setText(self._labels.name(lid))
            self._updating = False
            return
        # push to registry (will emit label_changed, but _on_label_changed_registry
        # will skip the text update since it's already correct)
        self._labels.set_name(lid, new_name)
        if lid == self._active_id:
            self._update_active_bar(lid)

    def _on_color_changed(self, label_id, color: QColor):
        self._labels.set_color(label_id, (color.red(), color.green(), color.blue()))
        if label_id == self._active_id:
            self._update_active_bar(label_id)

    def _on_visible_toggled(self, label_id, state):
        self._labels.set_visible(label_id, state == Qt.Checked)

    # ------------------------------------------------------------------
    # Row helpers
    # ------------------------------------------------------------------

    def _append_row(self, label_id):
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setRowHeight(row, 28)

        # COL_ID — read-only
        id_item = QTableWidgetItem(str(label_id))
        id_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)  # not editable
        id_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row, COL_ID, id_item)

        # COL_LABEL — editable
        name_item = QTableWidgetItem(self._labels.name(label_id))
        name_item.setFlags(
            Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        )
        self._table.setItem(row, COL_LABEL, name_item)

        # COL_COLOR — color button
        btn = _ColorButton(self._labels.qcolor(label_id, alpha=255))
        btn.color_changed.connect(lambda c, lid=label_id: self._on_color_changed(lid, c))
        self._table.setCellWidget(row, COL_COLOR, btn)

        # COL_VISIBLE — checkbox
        chk = QCheckBox()
        chk.setChecked(self._labels.is_visible(label_id))
        chk.setStyleSheet("QCheckBox { margin-left: 8px; }")
        chk.stateChanged.connect(lambda s, lid=label_id: self._on_visible_toggled(lid, s))
        self._table.setCellWidget(row, COL_VISIBLE, chk)

    def _row_for(self, label_id) -> int:
        """Return table row index for a label_id, or -1."""
        for row in range(self._table.rowCount()):
            item = self._table.item(row, COL_ID)
            if item and int(item.text()) == label_id:
                return row
        return -1

    def _label_id_at_row(self, row) -> int:
        item = self._table.item(row, COL_ID)
        if item is None:
            return None
        try:
            return int(item.text())
        except ValueError:
            return None

    def _select_row_for(self, label_id):
        row = self._row_for(label_id)
        if row >= 0:
            self._table.selectRow(row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_label(self):
        """Add a new label to the registry (triggers _on_label_added via signal)."""
        self._labels.add_label()

    def remove_selected(self):
        """Remove the currently selected label from registry and canvas mask."""
        row = self._table.currentRow()
        if row < 0:
            return
        lid = self._label_id_at_row(row)
        if lid is None:
            return
        # remove annotations for this label too
        self._annots.remove_by_label(lid)
        self._labels.remove_label(lid)   # triggers _on_label_removed via signal

    def active_label_id(self):
        return self._active_id

    def _update_active_bar(self, label_id):
        name = self._labels.name(label_id)
        qc   = self._labels.qcolor(label_id, alpha=255)
        fg   = "#000" if qc.lightness() > 140 else "#fff"
        self._active_bar.setText(
            "Active: <b>{}</b>  (ID {})".format(name, label_id)
        )
        self._active_bar.setStyleSheet(
            "QLabel {{"
            "  background: {bg}; color: {fg};"
            "  border-radius: 4px; padding: 5px; font-size: 11px;"
            "}}".format(bg=qc.name(), fg=fg)
        )

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log_state(self, event: str):
        """Log current labels and annotations after every panel-list change."""
        labels = self._labels.as_list()
        annots = self._annots.all()
        log.info(
            "[panel] %s — labels=%d  annotations=%d",
            event, len(labels), len(annots),
        )
        for lid, name, color in labels:
            r, g, b = color.red(), color.green(), color.blue()
            log.debug(
                "  label id=%-3d  name='%s'  color=(%d,%d,%d)%s",
                lid, name, r, g, b,
                "  [ACTIVE]" if lid == self._active_id else "",
            )
        for ann in annots:
            log.debug("  annotation %s", ann)

    # ------------------------------------------------------------------
    # Compat helpers used by window.py
    # ------------------------------------------------------------------

    def get_all(self):
        """[(label_id, name, QColor), ...] — same API as old LabelTable.get_all()."""
        return self._labels.as_list()

    def validate_label_ids(self):
        """Return (ok: bool, message: str) — same API as old LabelTable."""
        ids = self._labels.ids()
        if not ids:
            return False, "No labels defined."
        for lid in ids:
            if lid <= 0:
                return False, "Label ID must be > 0 (0 reserved for background)."
            if lid > 255:
                return False, "Label ID must be ≤ 255."
        return True, ""

    def active_color(self):
        if self._active_id:
            return self._labels.qcolor(self._active_id, alpha=255)
        return QColor(231, 76, 60)

    def active_name(self):
        return self._labels.name(self._active_id) if self._active_id else "Label 1"
