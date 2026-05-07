from .pg_panel import PgPanel
from .paint_view import PaintView
from .contrast_dialog import ContrastDialog
from .label_panel import LabelPanel
from ..data import (
    DEFAULT_HIGH_CUT,
    DEFAULT_LOW_CUT,
    build_label_id_mask,
    build_coco_annotation_json,
    compute_label_spectra,
)
from ..canvas import BgItem, CanvasItem
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QFileDialog,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QSplitter,
    QToolBar,
    QComboBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QColor, QImage, QPixmap, QBrush, QPainter, QPolygonF
from PyQt5.QtCore import QObject, QThread, QRectF, Qt, pyqtSignal, pyqtSlot, QPointF
import numpy as np
import os
import json
import logging
from ..registry import LabelRegistry, AnnotationRegistry

log = logging.getLogger(__name__)


class LabelSpectrumWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    @pyqtSlot(object, object, object)
    def process(self, datacube, mask, labels):
        try:
            self.finished.emit(compute_label_spectra(datacube, mask, labels))
        except Exception as exc:
            log.exception("LabelSpectrumWorker error")
            self.error.emit(str(exc))


class PaintWindow(QMainWindow):
    label_spectra_request = pyqtSignal(object, object, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSI Ground Truth Painter")
        self.resize(1500, 900)

        self._preview_low_cut = DEFAULT_LOW_CUT
        self._preview_high_cut = DEFAULT_HIGH_CUT

        # --- Registries (single source of truth) ---
        self._label_registry = LabelRegistry(self)
        self._annot_registry = AnnotationRegistry(self)

        # Seed two default labels
        self._label_registry.add_label("Label 1", (231,  76,  60))
        self._label_registry.add_label("Label 2", (46, 204, 113))

        # --- Canvas / scene ---
        self._scene = QGraphicsScene(self)
        self._bg = BgItem()
        self._canvas = CanvasItem()
        self._scene.addItem(self._bg)
        self._scene.addItem(self._canvas)
        self._scene.setSceneRect(QRectF(0, 0, 800, 600))

        self._view = PaintView(self._scene, self)
        self._pg_panel = PgPanel(self)

        # --- Label panel (left side) ---
        self._label_panel = LabelPanel(
            self._label_registry, self._annot_registry, self
        )
        self._label_panel.active_label_changed.connect(
            self._on_active_label_changed)

        # --- Layout: left panel | canvas view | right pg panel ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._label_panel)
        splitter.addWidget(self._view)
        splitter.addWidget(self._pg_panel)
        splitter.setSizes([230, 870, 400])
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(5)
        self.setCentralWidget(splitter)

        # --- Toolbar / statusbar ---
        self._tool_actions = {}
        self._bbox_items = []
        self._build_toolbar()
        self._build_statusbar()

        # --- Canvas signals ---
        self._canvas.signals.updated.connect(self._refresh_pg)
        self._canvas.signals.shape_closed.connect(self._compute_label_spectra)
        self._canvas.signals.spectrum_ready.connect(
            self._pg_panel.update_spectrum)
        self._canvas.signals.loaded.connect(self._on_loaded)

        # --- Registry → canvas sync ---
        self._label_registry.label_changed.connect(
            self._on_registry_label_changed)
        self._label_registry.label_color_changed.connect(
            self._on_registry_label_color_changed)
        self._label_registry.label_about_to_be_removed.connect(
            self._on_label_about_to_be_removed)
        self._label_registry.label_removed.connect(
            self._on_registry_label_removed)

        # --- Spectrum worker thread ---
        self._label_spectra_running = False
        self._label_spectra_pending = False
        self._spectrum_thread = QThread(self)
        self._spectrum_worker = LabelSpectrumWorker()
        self._spectrum_worker.moveToThread(self._spectrum_thread)
        self._spectrum_worker.finished.connect(self._on_label_spectra_ready)
        self._spectrum_worker.error.connect(self._on_label_spectra_error)
        self.label_spectra_request.connect(self._spectrum_worker.process)
        self._spectrum_thread.start()

        # Apply initial active label to canvas
        first_id = self._label_registry.ids()[0]
        self._apply_active_label(first_id)

    # ------------------------------------------------------------------
    # Registry → canvas sync
    # ------------------------------------------------------------------

    def _on_registry_label_changed(self, label_id):
        """When a label's name/color changes in registry, update canvas if active."""
        # Update active label if needed
        if label_id == self._label_panel.active_label_id():
            self._apply_active_label(label_id)

        # Respect visibility flag: if label toggled off -> hide pixels; toggled on -> restore
        vis = self._label_registry.is_visible(label_id)
        color = self._label_registry.color(label_id)
        if not vis:
            try:
                self._canvas.hide_label(label_id, color)
            except Exception:
                log.exception("Failed to hide label %s", label_id)
        else:
            try:
                self._canvas.show_label(label_id)
            except Exception:
                log.exception("Failed to show label %s", label_id)

        # Any label metadata change should refresh overlays so UI reflects the new state.
        self._refresh_pg()

    def _on_label_about_to_be_removed(self, label_id, old_color):
        """Fires BEFORE the label entry is deleted — erase its pixels from the mask."""
        if self._canvas is None or not self._canvas.is_loaded:
            return
        log.info("Erasing mask pixels for label id=%d  color=%s", label_id, old_color)
        self._canvas.erase_label_pixels(old_color)
        # erase_label_pixels emits updated + shape_closed, so refresh happens automatically

    def _on_registry_label_removed(self, label_id):
        """Pixels already erased by _on_label_about_to_be_removed; just switch active label."""
        log.info("Label removed from registry: id=%d", label_id)
        ids = self._label_registry.ids()
        if ids:
            self._apply_active_label(ids[0])
        else:
            # No labels left — clear active label on canvas so drawing is blocked
            log.info("No labels remaining — drawing disabled")
            self._canvas.set_current_label(None, "", QColor(0, 0, 0, 0))

    def _on_registry_label_color_changed(self, label_id, old_color, new_color):
        if self._canvas is None or not self._canvas.is_loaded:
            return
        self._canvas.recolor_label_pixels(old_color, new_color)
        self._refresh_pg()
        self._compute_label_spectra()

    # ------------------------------------------------------------------
    # Active label
    # ------------------------------------------------------------------

    def _on_active_label_changed(self, label_id):
        self._apply_active_label(label_id)
        self._compute_label_spectra()

    def _apply_active_label(self, label_id):
        name = self._label_registry.name(label_id)
        qc = self._label_registry.qcolor(label_id, alpha=220)
        self._canvas.set_pen_color(qc)
        self._canvas.set_current_label(label_id, name, qc)

    # ------------------------------------------------------------------
    # Toolbar label selector (QComboBox) — convenience for quick switch
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Canvas / pg events
    # ------------------------------------------------------------------

    def _on_loaded(self, ncols, nrows, nbands):
        log.info("Datacube loaded in window — %dx%d  bands=%d",
                 ncols, nrows, nbands)
        preview = self._format_preview_info()
        self.statusBar().showMessage(
            "Datacube loaded  |  {}×{} px  |  {} bands"
            "  |  {}  |  Ctrl+Wheel=Zoom  |  L=connect  |  Ctrl+S=save GT"
            .format(ncols, nrows, nbands, preview),
            0,
        )
        for action in self._tool_actions.values():
            action.setEnabled(True)
        self._contrast_action.setEnabled(True)

    def _refresh_pg(self):
        self._pg_panel.update_from_mask(self._canvas.get_mask())
        self._update_bbox_overlays()
        # Label spectra are computed only via shape_closed signal or explicit calls
        # (not on every mask refresh) to keep drawing responsive.

    def _compute_label_spectra(self):
        if self._canvas.datacube is None or self._canvas.is_drawing:
            return
        if self._label_spectra_running:
            self._label_spectra_pending = True
            return
        self._label_spectra_running = True
        self._label_spectra_pending = False
        self._pg_panel.set_spectrum_status("Computing label spectra...")
        self.label_spectra_request.emit(
            self._canvas.datacube,
            self._canvas.get_mask(),
            self._label_registry.as_list(),
        )

    def _on_label_spectra_ready(self, label_data):
        log.debug("Label spectra ready — %d class(es)", len(label_data))
        self._label_spectra_running = False
        self._pg_panel.update_class_spectra(label_data)
        if self._label_spectra_pending:
            self._compute_label_spectra()

    def _on_label_spectra_error(self, message):
        """Handle errors from the background label spectra worker."""
        log.error("Label spectra worker error: %s", message)
        self._label_spectra_running = False
        self._label_spectra_pending = False
        try:
            self._pg_panel.set_spectrum_status("Spectrum compute failed")
        except Exception:
            log.exception("Failed to update pg panel on spectra error")
        try:
            self.statusBar().showMessage(f"Spectra thread error: {message}", 5000)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _build_toolbar(self):
        tb = QToolBar("Tools", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        # Drawing tools
        for tool_key, label, shortcut in [
            ("connect", "Connect", "L"),
            ("circle",  "Circle",  "C"),
            ("eraser",  "Eraser",  "E"),
        ]:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setShortcut(shortcut)
            act.triggered.connect(lambda _, t=tool_key: self._set_tool(t))
            tb.addAction(act)
            self._tool_actions[tool_key] = act

        # Select Connect as default
        self._tool_actions["connect"].setChecked(True)
        tb.addSeparator()

        # Pen size
        tb.addWidget(QLabel("  Size: "))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(1, 100)
        self._size_spin.setValue(4)
        self._size_spin.setFixedWidth(60)
        self._size_spin.valueChanged.connect(self._canvas.set_pen_width)
        tb.addWidget(self._size_spin)

        # Opacity
        tb.addWidget(QLabel("  Opacity: "))
        opacity_spin = QSpinBox()
        opacity_spin.setRange(10, 100)
        opacity_spin.setValue(75)
        opacity_spin.setSuffix(" %")
        opacity_spin.setFixedWidth(70)
        opacity_spin.valueChanged.connect(
            lambda v: self._canvas.setOpacity(v / 100))
        tb.addWidget(opacity_spin)
        tb.addSeparator()

        # Zoom controls
        tb.addWidget(QLabel("  Zoom: "))
        for label, shortcut, slot in [
            ("+",  "Ctrl+=", self._view.zoom_in),
            ("-",  "Ctrl+-", self._view.zoom_out),
            ("1:1", "Ctrl+0", self._view.zoom_reset),
            ("Fit", "Ctrl+F", self._fit),
        ]:
            act = QAction(label, self)
            act.setShortcut(shortcut)
            act.triggered.connect(slot)
            tb.addAction(act)
        tb.addSeparator()

        # RGB Contrast
        self._contrast_action = QAction("RGB Contrast", self)
        self._contrast_action.triggered.connect(self._open_contrast_dialog)
        self._contrast_action.setEnabled(False)
        tb.addAction(self._contrast_action)
        tb.addSeparator()

        # File operations
        open_act = QAction("Open .hdr", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._open)
        tb.addAction(open_act)

        clear_act = QAction("Clear GT", self)
        clear_act.setShortcut("Ctrl+N")
        clear_act.triggered.connect(self._clear)
        tb.addAction(clear_act)

        save_act = QAction("Save GT", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._save)
        tb.addAction(save_act)
        # Load GT action
        load_gt_act = QAction("Load GT", self)
        load_gt_act.setShortcut("Ctrl+L")
        load_gt_act.triggered.connect(self._load_gt)
        tb.addAction(load_gt_act)

    def _build_statusbar(self):
        self._zoom_label = QLabel("Zoom: 100%")
        self.statusBar().addPermanentWidget(self._zoom_label)
        for action in self._tool_actions.values():
            action.setEnabled(False)
        self.statusBar().showMessage(
            "Please open a Hyperspectral Datacube (.hdr) first  |  Ctrl+O = open file"
        )

    # ------------------------------------------------------------------
    # Tool / view helpers
    # ------------------------------------------------------------------

    def update_zoom_label(self, zoom):
        self._zoom_label.setText("Zoom: {:.0f}%".format(zoom * 100))

    def _set_tool(self, tool_name):
        log.debug("Active tool: %s", tool_name)
        self._canvas.set_tool(tool_name)
        for name, action in self._tool_actions.items():
            action.setChecked(name == tool_name)

    def _fit(self):
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._view._zoom = self._view.transform().m11()
        self.update_zoom_label(self._view._zoom)

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Hyperspectral Datacube", "", "ENVI Header (*.hdr)"
        )
        if not path:
            return
        log.info("User opened: %s", path)
        self.statusBar().showMessage("Loading datacube...")
        QApplication.processEvents()
        self._canvas.set_preview_cuts(
            self._preview_low_cut, self._preview_high_cut)
        rgb_img = self._canvas.load_datacube(path)
        self._bg.setPixmap(QPixmap.fromImage(rgb_img))
        self._scene.setSceneRect(QRectF(rgb_img.rect()))

    def _load_gt(self):
        if not self._canvas.is_loaded:
            QMessageBox.warning(self, "No data", "Load a datacube before loading GT.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load COCO JSON", "", "COCO JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                coco = json.load(fh)
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", f"Failed to read JSON: {exc}")
            return
        images = coco.get("images", [])
        if not images:
            QMessageBox.warning(self, "Invalid COCO", "No images entry found in JSON")
            return
        image_entry = images[0]
        fname = image_entry.get("file_name", "")
        current_fname = os.path.basename(getattr(self._canvas.datacube, "filename", ""))
        if fname != current_fname:
            QMessageBox.warning(self, "Image mismatch", f"COCO image '{fname}' does not match current datacube '{current_fname}'")
            return

        # Clear existing labels and annotations
        self._label_registry.clear()
        self._annot_registry.clear()

        # Load categories into label registry
        categories = coco.get("categories", [])
        # deterministic color generator
        def _color_for(cid):
            return ((cid * 37) % 256, (cid * 67) % 256, (cid * 97) % 256)

        labels_data = {}
        for cat in categories:
            cid = int(cat.get("id", 0))
            name = str(cat.get("name", f"Label {cid}"))
            labels_data[cid] = {"name": name, "color": _color_for(cid), "visible": True}
        # registry.load expects dict with keys as ids
        self._label_registry.load({str(k): v for k, v in labels_data.items()})

        # Load annotations
        anns = coco.get("annotations", [])
        ann_list = []
        for a in anns:
            ann_list.append({
                "id": int(a.get("id", 0)),
                "label_id": int(a.get("category_id", 0)),
                "bbox": a.get("bbox", []),
                "segmentation": a.get("segmentation", []),
                "area": int(a.get("area", 0)),
            })
        self._annot_registry.load(ann_list)

        # Reconstruct mask from annotations (using QPainter)
        width = int(image_entry.get("width", getattr(self._canvas.datacube, "ncols", 0)))
        height = int(image_entry.get("height", getattr(self._canvas.datacube, "nrows", 0)))
        if width > 0 and height > 0:
            img = QImage(width, height, QImage.Format_ARGB32)
            img.fill(0)
            painter = QPainter(img)
            painter.setRenderHint(QPainter.Antialiasing)
            for ann in ann_list:
                lid = ann["label_id"]
                qc = self._label_registry.qcolor(lid, alpha=220)
                brush = QBrush(qc)
                painter.setBrush(brush)
                painter.setPen(Qt.NoPen)
                segs = ann.get("segmentation") or []
                if segs:
                    for seg in segs:
                        # seg is flat list [x1,y1,x2,y2,...]
                        pts = []
                        it = iter(seg)
                        for x, y in zip(it, it):
                            pts.append(QPointF(float(x), float(y)))
                        if len(pts) >= 3:
                            poly = QPolygonF(pts)
                            painter.drawPolygon(poly)
                else:
                    bbox = ann.get("bbox", [])
                    if len(bbox) >= 4:
                        x0, y0, w, h = bbox[:4]
                        painter.drawRect(int(x0), int(y0), int(w), int(h))
            painter.end()
            # assign mask to canvas
            self._canvas._mask = img
            self._canvas.setPixmap(QPixmap.fromImage(self._canvas._mask))
            self._canvas.signals.updated.emit()

        self.statusBar().showMessage(f"Loaded COCO annotations: {len(ann_list)} annotations, {len(labels_data)} categories", 6000)

    def _open_contrast_dialog(self):
        if not self._canvas.is_loaded:
            return
        orig_low, orig_high = self._preview_low_cut, self._preview_high_cut
        dialog = ContrastDialog(orig_low, orig_high, self)
        dialog.preview_changed.connect(self._apply_preview_cuts)
        if dialog.exec_() != QDialog.Accepted:
            self._apply_preview_cuts(orig_low, orig_high)
            self._preview_low_cut, self._preview_high_cut = orig_low, orig_high
            return
        self._preview_low_cut, self._preview_high_cut = dialog.values()
        self.statusBar().showMessage(
            "RGB contrast updated  |  {}".format(
                self._format_preview_info()), 5000
        )

    def _apply_preview_cuts(self, low_cut, high_cut):
        rgb_img = self._canvas.render_preview(low_cut, high_cut)
        if rgb_img:
            self._bg.setPixmap(QPixmap.fromImage(rgb_img))

    def _clear(self):
        if not self._canvas.is_loaded:
            return
        if QMessageBox.question(
            self, "Clear GT", "Clear the ground truth mask?",
            QMessageBox.Yes | QMessageBox.No,
        ) == QMessageBox.Yes:
            self._canvas.clear_mask()

    def _save(self):
        if not self._canvas.is_loaded:
            QMessageBox.warning(
                self, "No data", "Load a datacube before saving.")
            return
        ok, msg = self._label_panel.validate_label_ids()
        if not ok:
            QMessageBox.warning(self, "Invalid Label IDs", msg)
            return
        default_name = self._default_gt_filename()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save COCO Annotations", default_name,
            "COCO JSON (*.json)",
        )
        if not path:
            return

        labels = self._label_registry.as_list()

        # Always save as COCO JSON
        coco = build_coco_annotation_json(
            self._canvas.get_mask(), labels,
            image_id=1,
            file_name=os.path.basename(
                getattr(self._canvas.datacube, "filename", "")
            ),
        )
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(coco, fh, ensure_ascii=False, indent=2)
            ok = True
        except Exception:
            ok = False

        # Additionally save an 8-bit label mask PNG and an RGB preview PNG
        stem = os.path.splitext(path)[0]
        mask_path = f"{stem}_mask.png"
        rgb_path = f"{stem}_rgb.png"
        try:
            # Build label id mask and coerce to uint8 for viewing
            id_arr = build_label_id_mask(self._canvas.get_mask(), labels)
            id_arr8 = id_arr.astype(np.uint8)
            h, w = id_arr8.shape
            out_img = QImage(id_arr8.data, w, h, w, QImage.Format_Grayscale8).copy()
            out_img.save(mask_path)
            log.info("Saved 8-bit mask: %s", mask_path)

            # Also save a binary mask (0/1) to inspect occupied pixels
            id_bin = (id_arr > 0).astype(np.uint8)
            bin_path = f"{stem}_mask_bin.png"
            out_bin = QImage(id_bin.data, w, h, w, QImage.Format_Grayscale8).copy()
            out_bin.save(bin_path)
            log.info("Saved binary mask (0/1): %s", bin_path)
        except Exception:
            log.exception("Failed to save masks")

        try:
            rgb_img = self._canvas.render_preview(self._preview_low_cut, self._preview_high_cut)
            if rgb_img:
                rgb_img.save(rgb_path)
                log.info("Saved RGB preview: %s", rgb_path)
        except Exception:
            log.exception("Failed to save RGB preview")

        self.statusBar().showMessage(
            "{}:  {}  |  {} object(s)  |  COCO + mask + RGB saved".format(
                "Saved" if ok else "Save failed",
                path, len(coco.get("annotations", [])),
            ), 6000,
        )

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _format_preview_info(self):
        info = self._canvas.preview_info or {}
        low = info.get("low_cut",  self._preview_low_cut)
        high = info.get("high_cut", self._preview_high_cut)
        actual = info.get("actual_wavelengths")
        if actual:
            wl_text = "RGB~{:.0f}/{:.0f}/{:.0f} nm".format(*actual)
        else:
            bands = info.get("band_indices")
            wl_text = (
                "RGB bands={}/{}/{}".format(*bands)
                if bands else "RGB fallback bands"
            )
        return "{}  |  cut {:.1f}-{:.1f}%".format(wl_text, low, high)

    def _default_gt_filename(self):
        dc = self._canvas.datacube
        src = getattr(dc, "filename", "") if dc else ""
        name = os.path.basename(src)
        if not name:
            return "gt_mask"
        stem, ext = os.path.splitext(name)
        if ext.lower() == ".hdr":
            name = stem
        for suf in (".bip", ".bil", ".bsq"):
            if name.lower().endswith(suf):
                name = name[: -len(suf)]
                break
        return "{}".format(name or "gt_mask")

    def closeEvent(self, event):
        self._spectrum_thread.quit()
        self._spectrum_thread.wait(2000)
        super().closeEvent(event)
