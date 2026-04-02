import os
import json

import numpy as np
from PyQt5.QtCore import QObject, QThread, QRectF, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QFileDialog,
    QGraphicsScene,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QSplitter,
    QToolBar,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..canvas import BgItem, CanvasItem
from ..data import (
    DEFAULT_HIGH_CUT,
    DEFAULT_LOW_CUT,
    build_class_id_mask,
    build_coco_annotation_json,
    build_coco_annotation_json_from_layers,
    compute_class_spectra,
)
from .class_table import ClassTable


class ClassSpectrumWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    @pyqtSlot(object, object, object)
    def process(self, datacube, mask, classes):
        try:
            class_data = compute_class_spectra(
                datacube,
                mask,
                classes,
            )
            self.finished.emit(class_data)
        except Exception as e:
            self.error.emit(str(e))
from .contrast_dialog import ContrastDialog
from .paint_view import PaintView
from .pg_panel import PgPanel


class PaintWindow(QMainWindow):
    class_spectra_request = pyqtSignal(object, object, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSI Ground Truth Painter")
        self.resize(1800, 900)
        self._preview_low_cut = DEFAULT_LOW_CUT
        self._preview_high_cut = DEFAULT_HIGH_CUT

        self._scene = QGraphicsScene(self)
        self._bg = BgItem()
        self._canvas = CanvasItem()
        self._scene.addItem(self._bg)
        self._scene.addItem(self._canvas)
        self._scene.setSceneRect(QRectF(0, 0, 800, 600))

        self._view = PaintView(self._scene, self)
        self._pg_panel = PgPanel(self)
        self._class_table = ClassTable(self)
        self._class_table.setVisible(False)  # Hide class table from UI, use tags for class_id
        self._selected_class_id = self._class_table.active_class_id()
        self._selected_class_name = self._class_table.active_name()
        self._selected_class_color = self._class_table.active_color()

        self._layer_panel = QListWidget(self)
        self._layer_panel.setToolTip("Layer / Tag list (select to activate)")
        self._layer_panel.itemSelectionChanged.connect(self._on_layer_selection)

        layer_buttons = QWidget(self)
        layer_layout = QVBoxLayout(layer_buttons)
        layer_layout.setContentsMargins(4, 4, 4, 4)
        layer_layout.setSpacing(4)

        self._delete_layer_btn = QPushButton("🗑 delete", self)
        self._delete_layer_btn.clicked.connect(self._delete_layer)
        self._toggle_vis_btn = QPushButton("👁 visibility", self)
        self._toggle_vis_btn.clicked.connect(self._toggle_layer_visibility)
        self._toggle_lock_btn = QPushButton("🔒 lock", self)
        self._toggle_lock_btn.clicked.connect(self._toggle_layer_lock)

        layer_layout.addWidget(self._layer_panel)
        layer_layout.addWidget(self._delete_layer_btn)
        layer_layout.addWidget(self._toggle_vis_btn)
        layer_layout.addWidget(self._toggle_lock_btn)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._view)
        splitter.addWidget(self._pg_panel)
        splitter.addWidget(layer_buttons)
        splitter.setSizes([760, 520, 360])
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(5)
        self.setCentralWidget(splitter)

        self._build_toolbar()

        # Ensure tag label exists before any callback can trigger _update_tag_status
        self._tag_label = QLabel("Tag: -")

        self._build_statusbar()

        self._canvas.signals.updated.connect(self._refresh_pg)
        self._canvas.signals.updated.connect(self._refresh_layer_panel)
        self._canvas.signals.updated.connect(self._update_tag_status)
        self._canvas.signals.spectrum_ready.connect(self._pg_panel.update_spectrum)
        self._canvas.signals.loaded.connect(self._on_loaded)
        self._class_table.class_changed.connect(self._on_class_changed)

        self._class_spectra_running = False
        self._class_spectra_pending = False

        self._spectrum_thread = QThread(self)
        self._spectrum_worker = ClassSpectrumWorker()
        self._spectrum_worker.moveToThread(self._spectrum_thread)
        self._spectrum_worker.finished.connect(self._on_class_spectra_ready)
        self._spectrum_worker.error.connect(self._on_class_spectra_error)
        self.class_spectra_request.connect(self._spectrum_worker.process)
        self._spectrum_thread.start()

    def _on_loaded(self, ncols, nrows, nbands):
        preview = self._format_preview_info()
        self.statusBar().showMessage(
            "✅  Datacube โหลดสำเร็จ  │  {0}×{1} px  │  {2} bands"
            "  │  {3}  │  Ctrl+Wheel=Zoom  │  L=ต่อจุด  │  Ctrl+S=บันทึก GT".format(
                ncols, nrows, nbands, preview
            ),
            0,
        )
        for action in self._tool_actions.values():
            action.setEnabled(True)
        if hasattr(self, "_add_tag_action"):
            self._add_tag_action.setEnabled(True)
        self._contrast_action.setEnabled(True)
        self._refresh_layer_panel()
        self._update_tag_status()

    def _on_class_changed(self, class_id, name, color):
        self._selected_class_id = class_id
        self._selected_class_name = name
        self._selected_class_color = color
        self._refresh_class_selector()

        active_color = QColor(color)
        active_color.setAlpha(220)
        self._canvas.set_pen_color(active_color)
        self._color_preview.setStyleSheet(
            "background:{0}; border:2px solid #888; border-radius:4px;".format(
                color.name()
            )
        )
        self._set_tool("connect")
        self._compute_class_spectra()
        self._update_tag_status()

    def _delete_layer(self):
        selected = self._layer_panel.selectedItems()
        if not selected:
            return
        layer_id = selected[0].data(Qt.UserRole)
        self._canvas.remove_tag(layer_id)
        self._layer_panel.clearSelection()
        self._refresh_layer_panel()
        self._update_tag_status()

    def _toggle_layer_visibility(self):
        selected = self._layer_panel.selectedItems()
        if not selected:
            return
        layer_id = selected[0].data(Qt.UserRole)
        self._canvas.toggle_visibility(layer_id)
        self._refresh_layer_panel()
        self._update_tag_status()

    def _toggle_layer_lock(self):
        selected = self._layer_panel.selectedItems()
        if not selected:
            return
        layer_id = selected[0].data(Qt.UserRole)
        self._canvas.toggle_lock(layer_id)
        self._refresh_layer_panel()
        self._update_tag_status()

    def _update_tag_status(self):
        if not hasattr(self, "_tag_label") or self._tag_label is None:
            return

        layer = self._canvas.current_tag()
        if layer is None:
            self._tag_label.setText("Tag: -")
        else:
            self._tag_label.setText(
                f"Tag: {layer['id']} (class={layer['class_id']} name={layer['name']})"
            )

    def _refresh_class_selector(self):
        self._class_selector.blockSignals(True)
        self._class_selector.clear()
        selected_row = 0
        for idx, (class_id, name, color) in enumerate(self._class_table.get_all()):
            self._class_selector.addItem(f"{class_id}: {name}", (class_id, name, color))
            if class_id == getattr(self, "_selected_class_id", None):
                selected_row = idx

        if self._class_selector.count() > 0:
            if selected_row >= self._class_selector.count():
                selected_row = self._class_selector.count() - 1
            self._class_selector.setCurrentIndex(selected_row)
            self._on_selected_class_changed(selected_row)
        self._class_selector.blockSignals(False)

    def _on_selected_class_changed(self, index):
        if index < 0:
            return
        data = self._class_selector.itemData(index)
        if data is None:
            return
        class_id, name, color = data
        self._selected_class_id = class_id
        self._selected_class_name = name
        self._selected_class_color = color

        active_color = QColor(color)
        active_color.setAlpha(220)
        self._canvas.set_pen_color(active_color)
        self._canvas.set_current_class(class_id, name, active_color)

        if hasattr(self, "_color_preview") and self._color_preview is not None:
            self._color_preview.setStyleSheet(
                "background:{0}; border:2px solid #888; border-radius:4px;".format(
                    QColor(color).name()
                )
            )

        self._update_tag_status()

    def _refresh_layer_panel(self):
        self._layer_panel.blockSignals(True)
        self._layer_panel.clear()
        for layer in self._canvas.get_layers():
            visible = "✓" if layer.get("visible", True) else "✗"
            locked = "🔒" if layer.get("locked", False) else "🔓"
            text = f"ID={layer['id']} {layer['name']} class={layer['class_id']} {visible} {locked}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, layer["id"])
            self._layer_panel.addItem(item)
            if layer["id"] == self._canvas._active_layer_id:
                item.setSelected(True)
        self._layer_panel.blockSignals(False)

    def _on_layer_selection(self):
        selected = self._layer_panel.selectedItems()
        if not selected:
            return
        layer_id = selected[0].data(Qt.UserRole)
        if self._canvas.select_tag(layer_id):
            self._update_tag_status()

    def _add_class(self):
        # Add a new class through ClassTable dialog flow and refresh selector.
        new_class_id = self._class_table.add_class_dialog()
        if new_class_id is None:
            return
        self._selected_class_id = new_class_id
        self._refresh_class_selector()
        self.statusBar().showMessage("➕ สร้าง Class ใหม่เรียบร้อย", 2000)

    def _add_tag(self):
        if self._canvas.datacube is None:
            self.statusBar().showMessage("⚠️  โปรดโหลดไฟล์ก่อนสร้าง Tag", 3000)
            return

        class_id = getattr(self, "_selected_class_id", 1)
        name = getattr(self, "_selected_class_name", f"Tag")
        color = getattr(
            self, "_selected_class_color", QColor(231, 76, 60, 220)
        )

        layer_id = self._canvas.add_tag(class_id, name, color)
        if layer_id is not None:
            self.statusBar().showMessage(
                f"➕ สร้าง Tag ใหม่เรียบร้อย (ID={layer_id}, class={class_id})", 3000
            )
            self._refresh_layer_panel()
            self._update_tag_status()

    def _refresh_pg(self):
        self._pg_panel.update_from_mask(self._canvas.get_mask())
        self._compute_class_spectra()

    def _compute_class_spectra(self):
        if self._canvas.datacube is None or self._canvas.is_drawing:
            return
        if self._class_spectra_running:
            self._class_spectra_pending = True
            return

        self._class_spectra_running = True
        self._class_spectra_pending = False
        self._pg_panel.set_spectrum_status("Computing class spectra...")
        self.class_spectra_request.emit(
            self._canvas.datacube,
            self._canvas.get_mask(),
            self._class_table.get_all(),
        )

    def _on_class_spectra_ready(self, class_data):
        self._class_spectra_running = False
        self._pg_panel.update_class_spectra(class_data)
        if self._class_spectra_pending:
            self._compute_class_spectra()

    def _on_class_spectra_error(self, message):
        self._class_spectra_running = False
        self.statusBar().showMessage(f"Spectra thread error: {message}", 5000)

    def _build_toolbar(self):
        toolbar = QToolBar("Tools", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._class_selector = QComboBox(self)
        self._class_selector.setToolTip("Select class for new tags")
        self._class_selector.currentIndexChanged.connect(self._on_selected_class_changed)
        toolbar.addWidget(QLabel("Class:"))
        toolbar.addWidget(self._class_selector)
        self._color_preview = QLabel()
        self._color_preview.setFixedSize(21, 21)
        self._color_preview.setStyleSheet(
            "background:#e74c3c; border:2px solid #888; border-radius:4px;"
        )
        toolbar.addWidget(self._color_preview)

        act_add_class = QAction("➕ Class", self)
        act_add_class.setShortcut("Ctrl+Shift+C")
        act_add_class.triggered.connect(self._add_class)
        toolbar.addAction(act_add_class)

        act_cursor = QAction("⊚ Cursor", self)
        act_cursor.setCheckable(True)
        act_cursor.setShortcut("Ctrl+U")
        act_cursor.triggered.connect(lambda: self._set_tool("cursor"))
        toolbar.addAction(act_cursor)

        act_add_tag = QAction("➕ Tag", self)
        act_add_tag.setShortcut("T")
        act_add_tag.triggered.connect(self._add_tag)
        toolbar.addAction(act_add_tag)

        act_connect = QAction("🔗 ต่อจุด", self)
        act_connect.setCheckable(True)
        act_connect.setChecked(True)
        act_connect.setShortcut("L")
        act_connect.triggered.connect(lambda: self._set_tool("connect"))
        toolbar.addAction(act_connect)

        act_circle = QAction("⭕ วงกลม", self)
        act_circle.setCheckable(True)
        act_circle.setShortcut("C")
        act_circle.triggered.connect(lambda: self._set_tool("circle"))
        toolbar.addAction(act_circle)

        self._tool_actions = {
            "cursor": act_cursor,
            "connect": act_connect,
            "circle": act_circle,
        }
        self._add_tag_action = act_add_tag
        self._refresh_class_selector()
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("  🖌️ ขนาด:  "))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(1, 100)
        self._size_spin.setValue(4)
        self._size_spin.setFixedWidth(60)
        self._size_spin.valueChanged.connect(self._canvas.set_pen_width)
        toolbar.addWidget(self._size_spin)

        toolbar.addWidget(QLabel("  👁 Opacity:  "))
        opacity_spin = QSpinBox()
        opacity_spin.setRange(10, 100)
        opacity_spin.setValue(75)
        opacity_spin.setSuffix(" %")
        opacity_spin.setFixedWidth(70)
        opacity_spin.valueChanged.connect(lambda value: self._canvas.setOpacity(value / 100))
        toolbar.addWidget(opacity_spin)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("  🔍 Zoom:  "))
        zoom_in = QAction("＋", self)
        zoom_in.setShortcut("Ctrl+=")
        zoom_in.triggered.connect(self._view.zoom_in)
        toolbar.addAction(zoom_in)

        zoom_out = QAction("－", self)
        zoom_out.setShortcut("Ctrl+-")
        zoom_out.triggered.connect(self._view.zoom_out)
        toolbar.addAction(zoom_out)

        zoom_reset = QAction("1:1", self)
        zoom_reset.setShortcut("Ctrl+0")
        zoom_reset.triggered.connect(self._view.zoom_reset)
        toolbar.addAction(zoom_reset)

        fit_action = QAction("⊡ Fit", self)
        fit_action.setShortcut("Ctrl+F")
        fit_action.triggered.connect(self._fit)
        toolbar.addAction(fit_action)
        toolbar.addSeparator()

        self._contrast_action = QAction("🎚 RGB Contrast", self)
        self._contrast_action.triggered.connect(self._open_contrast_dialog)
        self._contrast_action.setEnabled(False)
        toolbar.addAction(self._contrast_action)
        toolbar.addSeparator()

        open_action = QAction("📂 เปิด .hdr", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open)
        toolbar.addAction(open_action)

        clear_action = QAction("🗋 ล้าง GT", self)
        clear_action.setShortcut("Ctrl+N")
        clear_action.triggered.connect(self._clear)
        toolbar.addAction(clear_action)

        save_action = QAction("💾 บันทึก GT", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save)
        toolbar.addAction(save_action)

    def _build_statusbar(self):
        self._zoom_label = QLabel("Zoom: 100%")
        self.statusBar().addPermanentWidget(self._zoom_label)
        self._tag_label = QLabel("Tag: -")
        self.statusBar().addPermanentWidget(self._tag_label)
        for action in self._tool_actions.values():
            action.setEnabled(False)
        if hasattr(self, "_add_tag_action"):
            self._add_tag_action.setEnabled(False)
        self.statusBar().showMessage(
            "⚠️  โปรดโหลด Hyperspectral Datacube (.hdr) ก่อน  │  Ctrl+O = เปิดไฟล์"
        )
        self._on_class_changed(
            self._class_table.active_class_id(),
            self._class_table.active_name(),
            self._class_table.active_color(),
        )

    def update_zoom_label(self, zoom):
        self._zoom_label.setText("Zoom: {0:.0f}%".format(zoom * 100))

    def _set_tool(self, tool_name):
        self._canvas.set_tool(tool_name)
        for name, action in self._tool_actions.items():
            action.setChecked(name == tool_name)

    def _fit(self):
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._view._zoom = self._view.transform().m11()
        self.update_zoom_label(self._view._zoom)

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "เปิด Hyperspectral Datacube", "", "ENVI Header (*.hdr)"
        )
        if not path:
            return
        self.statusBar().showMessage("⏳  กำลังโหลด datacube…")
        QApplication.processEvents()
        self._canvas.set_preview_cuts(self._preview_low_cut, self._preview_high_cut)
        rgb_img = self._canvas.load_datacube(path)
        self._bg.setPixmap(QPixmap.fromImage(rgb_img))
        self._scene.setSceneRect(QRectF(rgb_img.rect()))

    def _open_contrast_dialog(self):
        if not self._canvas.is_loaded:
            return
        original_low_cut = self._preview_low_cut
        original_high_cut = self._preview_high_cut
        dialog = ContrastDialog(original_low_cut, original_high_cut, self)
        dialog.preview_changed.connect(self._apply_preview_cuts)
        if dialog.exec_() != QDialog.Accepted:
            self._apply_preview_cuts(original_low_cut, original_high_cut)
            self._preview_low_cut = original_low_cut
            self._preview_high_cut = original_high_cut
            return
        self._preview_low_cut, self._preview_high_cut = dialog.values()
        self.statusBar().showMessage(
            "🎚 ปรับ RGB contrast แล้ว  │  {0}".format(self._format_preview_info()),
            5000,
        )

    def _apply_preview_cuts(self, low_cut, high_cut):
        rgb_img = self._canvas.render_preview(low_cut, high_cut)
        if rgb_img is None:
            return
        self._bg.setPixmap(QPixmap.fromImage(rgb_img))

    def _clear(self):
        if not self._canvas.is_loaded:
            return
        answer = QMessageBox.question(
            self, "ล้าง GT", "ล้าง Ground Truth layer?", QMessageBox.Yes | QMessageBox.No
        )
        if answer == QMessageBox.Yes:
            self._canvas.clear_mask()

    def _save(self):
        if not self._canvas.is_loaded:
            QMessageBox.warning(self, "ยังไม่มีข้อมูล", "โหลด datacube ก่อนบันทึก")
            return

        ok_ids, message = self._class_table.validate_class_ids()
        if not ok_ids:
            QMessageBox.warning(self, "ค่า Class ID ไม่ถูกต้อง", message)
            return

        default_name = self._default_gt_filename()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "บันทึก Ground Truth Mask หรือ COCO JSON",
            default_name,
            "COCO JSON (*.json);;PNG (*.png);;TIFF (*.tif *.tiff)",
        )
        if not path:
            return

        classes = self._class_table.get_all()

        if path.lower().endswith(".json"):
            layers = self._canvas.get_layers()
            if not layers:
                QMessageBox.warning(self, "ยังไม่มี Tag", "ยังไม่มี Tag ให้บันทึก")
                return

            coco = build_coco_annotation_json_from_layers(
                layers,
                classes=classes,
                image_id=1,
                file_name=os.path.basename(getattr(self._canvas.datacube, "filename", "")),
            )
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(coco, f, ensure_ascii=False, indent=2)
                ok = True
            except Exception:
                ok = False
            n_instances = len(coco.get("annotations", []))
            self.statusBar().showMessage(
                "{0}: {1}  │  {2} object(s)  │  COCO annotations".format(
                    "✅ บันทึกสำเร็จ" if ok else "❌ บันทึกไม่สำเร็จ",
                    path,
                    n_instances,
                ),
                6000,
            )
            return

        id_arr = build_class_id_mask(self._canvas.get_mask(), classes)
        height, width = id_arr.shape
        out_img = QImage(id_arr.data, width, height, width, QImage.Format_Grayscale8).copy()
        ok = out_img.save(path)
        n_classes = len([value for value in np.unique(id_arr) if value > 0])
        self.statusBar().showMessage(
            "{0}: {1}  │  {2} class(es)  │  pixel values = class ID (0=bg)".format(
                "✅ บันทึกสำเร็จ" if ok else "❌ บันทึกไม่สำเร็จ",
                path,
                n_classes,
            ),
            6000,
        )

    def _format_preview_info(self):
        preview_info = self._canvas.preview_info or {}
        low_cut = preview_info.get("low_cut", self._preview_low_cut)
        high_cut = preview_info.get("high_cut", self._preview_high_cut)
        actual = preview_info.get("actual_wavelengths")
        if actual:
            wavelength_text = "RGB≈{0:.0f}/{1:.0f}/{2:.0f} nm".format(*actual)
        else:
            bands = preview_info.get("band_indices")
            wavelength_text = (
                "RGB bands={0}/{1}/{2}".format(*bands)
                if bands
                else "RGB fallback bands"
            )
        return "{0}  │  cut {1:.1f}-{2:.1f}%".format(wavelength_text, low_cut, high_cut)

    def closeEvent(self, event):
        self._spectrum_thread.quit()
        self._spectrum_thread.wait(2000)
        super().closeEvent(event)

    def _default_gt_filename(self):
        datacube = self._canvas.datacube
        source_path = getattr(datacube, "filename", "") if datacube is not None else ""
        source_name = os.path.basename(source_path)
        if not source_name:
            return "gt_mask.png"

        name, ext = os.path.splitext(source_name)
        if ext.lower() == ".hdr":
            source_name = name

        lower_name = source_name.lower()
        for suffix in (".bip", ".bil", ".bsq"):
            if lower_name.endswith(suffix):
                source_name = source_name[: -len(suffix)]
                break

        if not source_name:
            source_name = "gt_mask"
        return "{0}.png".format(source_name)