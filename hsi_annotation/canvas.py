import time
import math
from collections import deque

import numpy as np
from PyQt5.QtCore import QObject, QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import QGraphicsPixmapItem

from .data import DEFAULT_HIGH_CUT, DEFAULT_LOW_CUT, build_rgb_preview, load_datacube_preview


class CanvasSignals(QObject):
    updated = pyqtSignal()
    spectrum_ready = pyqtSignal(int, int, object)
    loaded = pyqtSignal(int, int, int)


class BgItem(QGraphicsPixmapItem):
    def __init__(self):
        super().__init__()
        self.setZValue(0)
        self.setAcceptHoverEvents(False)

    def mousePressEvent(self, event):
        event.ignore()

    def mouseMoveEvent(self, event):
        event.ignore()

    def mouseReleaseEvent(self, event):
        event.ignore()


class CanvasItem(QGraphicsPixmapItem):
    def __init__(self):
        super().__init__()
        self.signals = CanvasSignals()
        self._datacube = None
        self._is_loaded = False
        self._drawing = False
        self._last_pos = QPointF()
        self._connect_start = None
        self._connect_last = None
        self._connect_points = []
        self._last_spectrum_emit = 0.0
        self._spectrum_interval_s = 0.06

        # Drag/move state
        self._dragging_bbox = False
        self._dragging_tag_id = None
        self._drag_start_pos = None
        self._drag_start_bbox = None
        self._drag_start_mask = None

        # Resize state
        self._resizing_bbox = False
        self._resize_layer_id = None
        self._resize_handle = None
        self._resize_start_pos = None
        self._resize_original_bbox = None   # bbox at drag-start
        self._resize_original_mask = None   # mask at drag-start
        self._hover_resize_zone = None

        self._current_class_id = 1
        self._current_class_name = "Class 1"
        self._current_class_color = QColor(231, 76, 60, 220)

        self._pen_color = QColor(231, 76, 60, 220)
        self._pen_width = 4
        self._tool = "connect"
        self._preview_low_cut = DEFAULT_LOW_CUT
        self._preview_high_cut = DEFAULT_HIGH_CUT
        self._preview_info = None
        self._circle_center = None
        self._circle_radius = 0
        self._layers = []
        self._active_layer_id = None
        self._next_layer_id = 1
        self._circle_base_mask = None
        self._init_mask(800, 600)
        self.setZValue(1)
        self.setOpacity(0.75)
        self.setShapeMode(QGraphicsPixmapItem.BoundingRectShape)
        self.setAcceptHoverEvents(True)

    # ------------------------------------------------------------------
    # paint() — one border per layer, drawn from bbox
    # ------------------------------------------------------------------

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        for layer in self._layers:
            if not layer.get("visible", True):
                continue
            bbox = layer.get("bbox")
            if not bbox:
                continue
            x, y, w, h = bbox

            # Border
            layer_color = QColor(layer.get("color", QColor(255, 0, 0)))
            layer_color.setAlpha(200)
            painter.setPen(QPen(layer_color, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x, y, w, h)

            # Label
            label = f"{layer.get('name', '')} ({layer.get('class_id', '')})"
            box_color = QColor(layer_color)
            box_color.setAlpha(180)
            painter.setBrush(box_color)
            painter.setPen(Qt.NoPen)
            painter.fillRect(x, y - 18, max(1, len(label) * 7), 18, box_color)
            painter.setPen(Qt.white)
            painter.drawText(x + 2, y - 4, label)

            # Resize handles (cursor mode only)
            if self._tool == "cursor":
                handle_color = (
                    QColor(255, 255, 255, 220)
                    if layer.get("id") == self._resize_layer_id
                    else QColor(200, 200, 200, 160)
                )
                painter.setBrush(QBrush(handle_color))
                painter.setPen(Qt.NoPen)
                hs = max(4, self._pen_width)
                for cx, cy in [
                    (x,         y        ),
                    (x + w,     y        ),
                    (x + w,     y + h    ),
                    (x,         y + h    ),
                    (x + w // 2, y       ),
                    (x + w // 2, y + h   ),
                    (x,          y + h // 2),
                    (x + w,      y + h // 2),
                ]:
                    painter.drawEllipse(QPointF(cx, cy), hs, hs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def datacube(self):
        return self._datacube

    @property
    def is_loaded(self):
        return self._is_loaded

    @property
    def is_drawing(self):
        return self._drawing

    @property
    def preview_info(self):
        return self._preview_info

    @property
    def preview_low_cut(self):
        return self._preview_low_cut

    @property
    def preview_high_cut(self):
        return self._preview_high_cut

    # ------------------------------------------------------------------
    # Init / setup
    # ------------------------------------------------------------------

    def _init_mask(self, width, height):
        self._mask = QImage(width, height, QImage.Format_ARGB32)
        self._mask.fill(0)
        for layer in self._layers:
            layer["mask"] = QImage(width, height, QImage.Format_ARGB32)
            layer["mask"].fill(0)
        self.setPixmap(QPixmap.fromImage(self._mask))

    def set_tool(self, tool_name):
        self._tool = tool_name
        if tool_name != "connect":
            self._connect_start = None
            self._connect_last = None
            self._connect_points = []
        if tool_name != "cursor":
            self._resize_layer_id = None
            self._resizing_bbox = False
            self._resize_handle = None
            self._resize_start_pos = None
            self._resize_original_bbox = None
            self._resize_original_mask = None

    def set_pen_color(self, color):
        self._pen_color = color

    def set_pen_width(self, width):
        self._pen_width = width

    def set_preview_cuts(self, low_cut, high_cut):
        self._preview_low_cut = float(low_cut)
        self._preview_high_cut = float(high_cut)

    def get_mask(self):
        return self._mask

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def add_tag(self, class_id, name, color):
        if self._mask is None:
            return None
        width = self._mask.width()
        height = self._mask.height()
        layer = {
            "id": self._next_layer_id,
            "class_ids": [class_id],
            "class_id": class_id,
            "name": name,
            "color": QColor(color),
            "visible": True,
            "locked": False,
            "mask": QImage(width, height, QImage.Format_ARGB32),
            # bbox = tight bounding box of painted pixels (auto-updated after each stroke)
            # None until the user paints something
            "bbox": None,
            "area": 0,
        }
        layer["mask"].fill(0)
        self._layers.append(layer)
        self._active_layer_id = layer["id"]
        self._next_layer_id += 1
        self._compose_mask()
        self.signals.updated.emit()
        return layer["id"]

    def select_tag(self, layer_id):
        if any(layer["id"] == layer_id for layer in self._layers):
            self._active_layer_id = layer_id
            return True
        return False

    def current_tag(self):
        for layer in self._layers:
            if layer["id"] == self._active_layer_id:
                return layer
        return None

    def get_layers(self):
        return self._layers

    def remove_tag(self, layer_id):
        self._layers = [l for l in self._layers if l["id"] != layer_id]
        if self._active_layer_id == layer_id:
            self._active_layer_id = self._layers[-1]["id"] if self._layers else None
        self._compose_mask()
        self.signals.updated.emit()

    def toggle_visibility(self, layer_id):
        for layer in self._layers:
            if layer["id"] == layer_id:
                layer["visible"] = not layer.get("visible", True)
                break
        self._compose_mask()
        self.signals.updated.emit()

    def toggle_lock(self, layer_id):
        for layer in self._layers:
            if layer["id"] == layer_id:
                layer["locked"] = not layer.get("locked", False)
                break
        self.signals.updated.emit()

    def clear_mask(self):
        self._mask.fill(0)
        for layer in self._layers:
            layer["mask"].fill(0)
        self._layers = []
        self._active_layer_id = None
        self.setPixmap(QPixmap.fromImage(self._mask))
        self.signals.updated.emit()

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def set_current_class(self, class_id, name, color):
        self._current_class_id = class_id
        self._current_class_name = name
        self._current_class_color = color
        self._pen_color = QColor(color)
        self._pen_color.setAlpha(220)

    def _ensure_active_tag_for_current_class(self):
        current_tag = self.current_tag()
        if current_tag is None or current_tag.get("class_id") != self._current_class_id:
            self.add_tag(self._current_class_id, self._current_class_name, self._current_class_color)

    def _compose_mask(self):
        if self._mask is None:
            return
        self._mask.fill(0)
        painter = QPainter(self._mask)
        painter.setRenderHint(QPainter.Antialiasing)
        for layer in self._layers:
            if not layer.get("visible", True):
                continue
            painter.drawImage(0, 0, layer["mask"])
        painter.end()
        self.setPixmap(QPixmap.fromImage(self._mask))

    # ------------------------------------------------------------------
    # Hit-testing & cursor
    # ------------------------------------------------------------------

    def _layer_at_point(self, pos):
        """Return topmost layer whose bbox contains pos."""
        for layer in reversed(self._layers):
            bbox = layer.get("bbox")
            if not bbox or not layer.get("visible", True):
                continue
            x, y, w, h = bbox
            if x <= pos.x() <= x + w and y <= pos.y() <= y + h:
                return layer
        return None

    def _find_resize_zone(self, pos, bbox, threshold=8):
        """Return handle name for edges/corners of bbox, or None."""
        if bbox is None:
            return None
        x, y, w, h = bbox
        right, bottom = x + w, y + h
        if not (x - threshold <= pos.x() <= right  + threshold and
                y - threshold <= pos.y() <= bottom + threshold):
            return None
        nl = abs(pos.x() - x)      <= threshold
        nr = abs(pos.x() - right)  <= threshold
        nt = abs(pos.y() - y)       <= threshold
        nb = abs(pos.y() - bottom)  <= threshold
        if nt and nl: return "nw"
        if nt and nr: return "ne"
        if nb and nr: return "se"
        if nb and nl: return "sw"
        if nl and y + threshold < pos.y() < bottom - threshold: return "w"
        if nr and y + threshold < pos.y() < bottom - threshold: return "e"
        if nt and x + threshold < pos.x() < right  - threshold: return "n"
        if nb and x + threshold < pos.x() < right  - threshold: return "s"
        return None

    def _update_hover_cursor(self, pos):
        if self._tool != "cursor":
            self.setCursor(Qt.ArrowCursor)
            return
        layer = self._layer_at_point(pos)
        if layer is None:
            self._hover_resize_zone = None
            self._resize_layer_id = None
            self.setCursor(Qt.ArrowCursor)
            return
        zone = self._find_resize_zone(pos, layer.get("bbox"))
        self._hover_resize_zone = zone
        if zone is None:
            self._resize_layer_id = None
            self.setCursor(Qt.SizeAllCursor)
            return
        self._resize_layer_id = layer.get("id")
        cursor_map = {
            "w":  Qt.SizeHorCursor,   "e":  Qt.SizeHorCursor,
            "n":  Qt.SizeVerCursor,   "s":  Qt.SizeVerCursor,
            "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor,
            "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor,
        }
        self.setCursor(cursor_map.get(zone, Qt.ArrowCursor))

    # ------------------------------------------------------------------
    # Resize / move computation
    # ------------------------------------------------------------------

    def _compute_new_bbox(self, cur_pos, orig_bbox, handle):
        """Return new [x, y, w, h] given cursor, original bbox, and handle."""
        x0, y0, w0, h0 = orig_bbox
        dx = int(cur_pos.x() - self._resize_start_pos.x())
        dy = int(cur_pos.y() - self._resize_start_pos.y())
        x, y, w, h = x0, y0, w0, h0
        if handle in ("w", "nw", "sw"):
            w = max(1, w0 - dx);  x = x0 + (w0 - w)
        elif handle in ("e", "ne", "se"):
            w = max(1, w0 + dx)
        if handle in ("n", "nw", "ne"):
            h = max(1, h0 - dy);  y = y0 + (h0 - h)
        elif handle in ("s", "sw", "se"):
            h = max(1, h0 + dy)
        x = max(0, x);  y = max(0, y)
        w = min(self._mask.width()  - x, w)
        h = min(self._mask.height() - y, h)
        return [x, y, w, h]

    # ------------------------------------------------------------------
    # Mask transform helpers
    # ------------------------------------------------------------------

    def _scale_mask(self, src_mask, old_bbox, new_bbox):
        """Scale painted region from old_bbox to new_bbox."""
        ox, oy, ow, oh = old_bbox
        nx, ny, nw, nh = new_bbox
        src_cropped = src_mask.copy(ox, oy, max(1, ow), max(1, oh))
        scaled = src_cropped.scaled(
            max(1, nw), max(1, nh),
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation,
        )
        new_mask = QImage(src_mask.width(), src_mask.height(), QImage.Format_ARGB32)
        new_mask.fill(0)
        p = QPainter(new_mask)
        p.drawImage(nx, ny, scaled)
        p.end()
        return new_mask

    def _offset_mask(self, mask, dx, dy):
        new_mask = QImage(mask.size(), QImage.Format_ARGB32)
        new_mask.fill(0)
        p = QPainter(new_mask)
        p.drawImage(dx, dy, mask)
        p.end()
        return new_mask

    def _update_bbox_from_pixels(self, layer):
        """Auto-recompute layer['bbox'] as tight bounding box of painted pixels."""
        mask = layer.get("mask")
        if mask is None:
            layer["bbox"] = None
            layer["area"] = 0
            return
        mask_rgba = mask.convertToFormat(QImage.Format_RGBA8888)
        cw, ch = mask_rgba.width(), mask_rgba.height()
        ptr = mask_rgba.bits()
        ptr.setsize(ch * cw * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((ch, cw, 4)).copy()
        alpha = arr[:, :, 3]
        ys, xs = np.where(alpha > 0)
        if xs.size == 0:
            layer["bbox"] = None
            layer["area"] = 0
            return
        layer["bbox"] = [
            int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ]
        layer["area"] = int((alpha > 0).sum())

    def _update_active_bbox(self):
        layer = self.current_tag()
        if layer is not None:
            self._update_bbox_from_pixels(layer)

    # ------------------------------------------------------------------
    # Datacube / preview
    # ------------------------------------------------------------------

    def load_datacube(self, path):
        self._datacube, rgb_img, self._preview_info = load_datacube_preview(
            path, low_cut=self._preview_low_cut, high_cut=self._preview_high_cut,
        )
        self._init_mask(rgb_img.width(), rgb_img.height())
        self._is_loaded = True
        self.signals.loaded.emit(rgb_img.width(), rgb_img.height(), self._datacube.nbands)
        return rgb_img

    def render_preview(self, low_cut=None, high_cut=None):
        if self._datacube is None:
            return None
        if low_cut is not None:
            self._preview_low_cut = float(low_cut)
        if high_cut is not None:
            self._preview_high_cut = float(high_cut)
        rgb, self._preview_info = build_rgb_preview(
            self._datacube, low_cut=self._preview_low_cut, high_cut=self._preview_high_cut,
        )
        rgb8 = np.ascontiguousarray((rgb * 255).clip(0, 255).astype(np.uint8))
        height, width, _ = rgb8.shape
        return QImage(
            rgb8.data, width, height, width * 3, QImage.Format_RGB888
        ).convertToFormat(QImage.Format_ARGB32).copy()

    def _emit_spectrum(self, pos, force=False):
        if self._datacube is None:
            return
        now = time.perf_counter()
        if not force and (now - self._last_spectrum_emit) < self._spectrum_interval_s:
            return
        x, y = int(pos.x()), int(pos.y())
        if 0 <= x < self._datacube.ncols and 0 <= y < self._datacube.nrows:
            spec = np.array(self._datacube[y, x, :], dtype=np.float32).flatten()
            self._last_spectrum_emit = now
            self.signals.spectrum_ready.emit(x, y, spec)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if not self._is_loaded:
            return
        if event.button() == Qt.LeftButton:
            if self._tool in ("fill", "connect", "circle", "pen", "eraser"):
                self._ensure_active_tag_for_current_class()

            if self._tool == "fill":
                self.flood_fill(event.pos())

            elif self._tool == "connect":
                self._connect_click(event.pos())
                self._emit_spectrum(event.pos(), force=True)

            elif self._tool == "circle":
                self._drawing = True
                self._circle_center = event.pos()
                self._circle_radius = 0
                self._circle_base_mask = self._mask.copy()
                self._draw_circle_preview(0)
                self._emit_spectrum(event.pos(), force=True)

            elif self._tool == "cursor":
                layer = self._layer_at_point(event.pos())
                zone = self._find_resize_zone(
                    event.pos(), layer.get("bbox") if layer else None
                ) if layer else None

                if layer and zone is not None:
                    # Start resize — save original bbox + mask
                    self._resizing_bbox = True
                    self._resize_handle = zone
                    self._resize_layer_id = layer.get("id")
                    self._resize_start_pos = event.pos()
                    self._resize_original_bbox = list(layer["bbox"])
                    self._resize_original_mask = layer["mask"].copy()
                    self.signals.updated.emit()
                    return

                if layer:
                    # Start move
                    self._dragging_bbox = True
                    self._dragging_tag_id = layer.get("id")
                    self._drag_start_pos = event.pos()
                    self._drag_start_bbox = list(layer["bbox"])
                    self._drag_start_mask = layer["mask"].copy()

                self._emit_spectrum(event.pos(), force=True)

            else:
                self._drawing = True
                self._last_pos = event.pos()
                self._draw_dot(event.pos())
                self._emit_spectrum(event.pos(), force=True)

        elif event.button() == Qt.RightButton and self._tool == "connect":
            self._close_connect_path()

    def mouseDoubleClickEvent(self, event):
        event.ignore()

    def hoverMoveEvent(self, event):
        if not self._is_loaded:
            return
        self._update_hover_cursor(event.pos())
        self.signals.updated.emit()

    def mouseMoveEvent(self, event):
        if not self._is_loaded:
            return

        if self._tool == "circle":
            if self._drawing and (event.buttons() & Qt.LeftButton):
                dx = event.pos().x() - self._circle_center.x()
                dy = event.pos().y() - self._circle_center.y()
                self._circle_radius = int(math.hypot(dx, dy))
                self._draw_circle_preview(self._circle_radius)
            return

        if self._tool == "cursor":
            if self._resizing_bbox and self._resize_handle is not None:
                layer = next(
                    (l for l in self._layers if l.get("id") == self._resize_layer_id), None
                )
                if layer is not None:
                    # Live: update bbox and scale mask each frame
                    new_bbox = self._compute_new_bbox(
                        event.pos(), self._resize_original_bbox, self._resize_handle
                    )
                    layer["bbox"] = new_bbox
                    if self._resize_original_mask is not None:
                        layer["mask"] = self._scale_mask(
                            self._resize_original_mask,
                            self._resize_original_bbox,
                            new_bbox,
                        )
                    self._compose_mask()
                self.signals.updated.emit()
                return

            if self._dragging_bbox and self._dragging_tag_id is not None:
                dx = int(event.pos().x() - self._drag_start_pos.x())
                dy = int(event.pos().y() - self._drag_start_pos.y())
                layer = next(
                    (l for l in self._layers if l.get("id") == self._dragging_tag_id), None
                )
                if layer is not None and self._drag_start_bbox is not None:
                    x0, y0, w0, h0 = self._drag_start_bbox
                    layer["bbox"] = [x0 + dx, y0 + dy, w0, h0]
                    if self._drag_start_mask is not None:
                        layer["mask"] = self._offset_mask(self._drag_start_mask, dx, dy)
                    self._compose_mask()
                self._emit_spectrum(event.pos())
                return

            self._update_hover_cursor(event.pos())
            self._emit_spectrum(event.pos())
            return

        if self._tool in ("fill", "connect"):
            return

        if self._drawing and (event.buttons() & Qt.LeftButton):
            self._draw_line(self._last_pos, event.pos())
            self._last_pos = event.pos()
            self._emit_spectrum(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._tool in ("pen", "eraser"):
                self._drawing = False
                self.signals.updated.emit()

            elif self._tool == "circle":
                if self._drawing:
                    self._paint_circle(self._circle_center, self._circle_radius)
                self._drawing = False
                self._circle_center = None
                self._circle_radius = 0
                self._circle_base_mask = None
                self.signals.updated.emit()

            elif self._tool == "cursor":
                if self._resizing_bbox:
                    # Resize done — recalculate bbox tightly from painted pixels
                    layer = next(
                        (l for l in self._layers if l.get("id") == self._resize_layer_id), None
                    )
                    if layer is not None:
                        self._update_bbox_from_pixels(layer)
                    self._resizing_bbox = False
                    self._resize_handle = None
                    self._resize_start_pos = None
                    self._resize_original_bbox = None
                    self._resize_original_mask = None
                    self.signals.updated.emit()
                    return

                if self._dragging_bbox:
                    # Move done — recalculate bbox from pixels
                    layer = next(
                        (l for l in self._layers if l.get("id") == self._dragging_tag_id), None
                    )
                    if layer is not None:
                        self._update_bbox_from_pixels(layer)

                self._dragging_bbox = False
                self._dragging_tag_id = None
                self._drag_start_pos = None
                self._drag_start_bbox = None
                self._drag_start_mask = None
                self.signals.updated.emit()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _ensure_layer_mask(self, layer):
        if layer is None:
            return False
        if layer.get("mask") is None:
            layer["mask"] = QImage(
                self._mask.width(), self._mask.height(), QImage.Format_RGBA8888
            )
            layer["mask"].fill(QColor(0, 0, 0, 0))
        return True

    def _make_pen(self):
        if self._tool == "eraser":
            return QPen(Qt.transparent, self._pen_width * 3,
                        Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        color = QColor(self._pen_color)
        color.setAlpha(220)
        return QPen(color, self._pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

    def _paint_on_layer(self, painter_fn):
        """Paint onto the active layer mask. No clipping — full canvas is the canvas."""
        layer = self.current_tag()
        if layer is None or layer.get("locked", False):
            return
        if not self._ensure_layer_mask(layer):
            return
        mask = layer.get("mask")
        painter = QPainter(mask)
        if self._tool == "eraser":
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._tool != "circle":
            painter.setPen(self._make_pen())
        painter_fn(painter)
        painter.end()
        self._compose_mask()

    def _draw_dot(self, pos):
        self._paint_on_layer(lambda p: p.drawPoint(pos))
        self._update_active_bbox()

    def _draw_line(self, p1, p2):
        self._paint_on_layer(lambda p: p.drawLine(p1, p2))
        self._update_active_bbox()

    def _paint_circle(self, center, radius):
        if center is None or radius <= 0:
            return
        layer = self.current_tag()
        if layer is None or layer.get("locked", False):
            return
        if not self._ensure_layer_mask(layer):
            return
        mask = layer.get("mask")
        painter = QPainter(mask)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        color = QColor(self._pen_color)
        color.setAlpha(220)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(center, radius, radius)
        painter.end()
        self._compose_mask()
        self._update_active_bbox()

    def _draw_circle_preview(self, radius):
        if self._circle_center is None or self._circle_base_mask is None:
            return
        preview_mask = self._circle_base_mask.copy()
        painter = QPainter(preview_mask)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        color = QColor(self._pen_color)
        color.setAlpha(220)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(self._circle_center, radius, radius)
        painter.end()
        self.setPixmap(QPixmap.fromImage(preview_mask))

    def _connect_click(self, pos):
        tag = self.current_tag()
        if tag is None or tag.get("locked", False):
            return
        if self._connect_last is None and self._connect_points:
            self._connect_points = []
            self._connect_start = None
        point = QPointF(pos)
        if self._connect_last is None:
            self._connect_start = QPointF(point)
            self._connect_last = QPointF(point)
            self._connect_points = [QPointF(point)]
            self._draw_dot(point)
            self.signals.updated.emit()
            return
        if len(self._connect_points) >= 3:
            dx = point.x() - self._connect_start.x()
            dy = point.y() - self._connect_start.y()
            if math.hypot(dx, dy) <= max(5.0, self._pen_width):
                self._draw_line(self._connect_last, self._connect_start)
                self._connect_points.append(QPointF(self._connect_start))
                self._fill_connect_polygon()
                self._connect_start = None
                self._connect_last = None
                self._connect_points = []
                self.signals.updated.emit()
                return
        self._draw_line(self._connect_last, point)
        self._connect_last = QPointF(point)
        self._connect_points.append(QPointF(point))
        self.signals.updated.emit()

    def _fill_connect_polygon(self):
        if len(self._connect_points) < 3:
            return
        points = QPolygonF(self._connect_points)
        def fill_polygon(painter):
            color = QColor(self._pen_color)
            color.setAlpha(220)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawPolygon(points)
        self._paint_on_layer(fill_polygon)
        self._update_active_bbox()

    def _close_connect_path(self):
        tag = self.current_tag()
        if tag is None or tag.get("locked", False):
            return
        if self._connect_last is None or self._connect_start is None:
            self._connect_start = None
            self._connect_last = None
            self._connect_points = []
            return
        if self._connect_last != self._connect_start:
            self._draw_line(self._connect_last, self._connect_start)
        if len(self._connect_points) >= 3:
            self._fill_connect_polygon()
        self._connect_start = None
        self._connect_last = None
        self._connect_points = []
        self.signals.updated.emit()

    # ------------------------------------------------------------------
    # Flood fill
    # ------------------------------------------------------------------

    def flood_fill(self, pos):
        x0, y0 = int(pos.x()), int(pos.y())
        image = self._mask.convertToFormat(QImage.Format_RGBA8888)
        width, height = image.width(), image.height()
        if not (0 <= x0 < width and 0 <= y0 < height):
            return
        ptr = image.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4)).copy()
        target_color = tuple(arr[y0, x0])
        color = self._pen_color
        fill_color = np.array([color.red(), color.green(), color.blue(), 220], dtype=np.uint8)
        if target_color == tuple(fill_color):
            return
        queue = deque([(y0, x0)])
        while queue:
            y, x = queue.popleft()
            if tuple(arr[y, x]) != target_color:
                continue
            arr[y, x] = fill_color
            if x > 0:           queue.append((y, x - 1))
            if x < width - 1:   queue.append((y, x + 1))
            if y > 0:           queue.append((y - 1, x))
            if y < height - 1:  queue.append((y + 1, x))
        self._mask = QImage(
            arr.data, width, height, width * 4, QImage.Format_RGBA8888
        ).copy().convertToFormat(QImage.Format_ARGB32)
        self.setPixmap(QPixmap.fromImage(self._mask))
        self.signals.updated.emit()
