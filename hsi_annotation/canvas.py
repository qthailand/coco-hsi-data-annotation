import logging
import time
import math
from collections import deque

import numpy as np
from PyQt5.QtCore import QObject, QPointF, QRunnable, QThreadPool, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import QGraphicsPixmapItem

from .data import (
    DEFAULT_HIGH_CUT,
    DEFAULT_LOW_CUT,
    _match_color,
    build_rgb_preview,
    load_datacube_preview,
)

log = logging.getLogger(__name__)


class _SpecSignals(QObject):
    ready = pyqtSignal(int, int, object)


class _SpectrumRunnable(QRunnable):
    """Read a single pixel spectrum in a thread-pool worker."""
    def __init__(self, datacube, x, y):
        super().__init__()
        self.signals = _SpecSignals()
        self._datacube = datacube
        self._x = x
        self._y = y

    def run(self):
        try:
            spec = np.array(self._datacube[self._y, self._x, :], dtype=np.float32).flatten()
            self.signals.ready.emit(self._x, self._y, spec)
        except Exception:
            log.exception("SpectrumRunnable: failed to read pixel (%d, %d)", self._x, self._y)


class CanvasSignals(QObject):
    updated = pyqtSignal()
    shape_closed = pyqtSignal()   # emitted when a shape is finalised (connect closed / circle right-release / eraser up / clear)
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

        self._current_label_id = 1
        self._current_label_name = "Label 1"
        self._current_label_color = QColor(231, 76, 60, 220)

        self._pen_color = QColor(231, 76, 60, 220)
        self._pen_width = 4
        self._tool = "connect"
        self._preview_low_cut = DEFAULT_LOW_CUT
        self._preview_high_cut = DEFAULT_HIGH_CUT
        self._preview_info = None
        self._circle_center = None
        self._circle_radius = 0
        self._circle_base_mask = None
        # Store per-label hidden pixel fragments for temporary hide/show
        # key: label_id -> ndarray of shape (h,w,4) containing only the stored pixels
        self._hidden_label_masks = {}
        self._init_mask(800, 600)
        self.setZValue(1)
        self.setOpacity(0.75)
        self.setShapeMode(QGraphicsPixmapItem.BoundingRectShape)
        self.setAcceptHoverEvents(True)

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

    @property
    def has_active_label(self):
        """True when a label is selected and drawing is permitted."""
        return self._current_label_id is not None

    # ------------------------------------------------------------------
    # Init / setup
    # ------------------------------------------------------------------

    def _init_mask(self, width, height):
        self._mask = QImage(width, height, QImage.Format_ARGB32)
        self._mask.fill(0)
        self.setPixmap(QPixmap.fromImage(self._mask))

    def set_tool(self, tool_name):
        log.debug("Tool changed: %s", tool_name)
        self._tool = tool_name
        self._drawing = False
        if tool_name != "connect":
            self._connect_start = None
            self._connect_last = None
            self._connect_points = []

    def set_pen_color(self, color):
        self._pen_color = color

    def set_pen_width(self, width):
        self._pen_width = width

    def set_preview_cuts(self, low_cut, high_cut):
        self._preview_low_cut = float(low_cut)
        self._preview_high_cut = float(high_cut)

    def get_mask(self):
        return self._mask

    def set_current_label(self, label_id, name, color):
        """
        Set the active label.  Pass label_id=None to indicate "no label" —
        drawing tools will be blocked until a valid label is set again.
        """
        self._current_label_id = label_id
        self._current_label_name = name
        if label_id is None:
            log.info("Active label cleared — drawing disabled until a label is selected")
            self._pen_color = QColor(0, 0, 0, 0)
        else:
            self._current_label_color = color
            self._pen_color = QColor(color)
            self._pen_color.setAlpha(220)

    def clear_mask(self):
        self._mask.fill(0)
        self.setPixmap(QPixmap.fromImage(self._mask))
        self.signals.updated.emit()
        self.signals.shape_closed.emit()

    # ------------------------------------------------------------------
    # Datacube / preview
    # ------------------------------------------------------------------

    def load_datacube(self, path):
        log.info("CanvasItem.load_datacube: %s", path)
        self._datacube, rgb_img, self._preview_info = load_datacube_preview(
            path, low_cut=self._preview_low_cut, high_cut=self._preview_high_cut,
        )
        self._init_mask(rgb_img.width(), rgb_img.height())
        self._is_loaded = True
        log.info("Datacube loaded — %dx%d px  %d bands", rgb_img.width(), rgb_img.height(), self._datacube.nbands)
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
        """Schedule a pixel-spectrum read on a thread-pool worker to avoid blocking the UI."""
        if self._datacube is None:
            return
        now = time.perf_counter()
        if not force and (now - self._last_spectrum_emit) < self._spectrum_interval_s:
            return
        x, y = int(pos.x()), int(pos.y())
        if 0 <= x < self._datacube.ncols and 0 <= y < self._datacube.nrows:
            self._last_spectrum_emit = now
            worker = _SpectrumRunnable(self._datacube, x, y)
            # QueuedConnection ensures the result is delivered on the main-thread event loop
            worker.signals.ready.connect(self.signals.spectrum_ready, Qt.QueuedConnection)
            QThreadPool.globalInstance().start(worker)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if not self._is_loaded:
            return
        if event.button() == Qt.LeftButton:
            # Eraser works without a label; all paint tools require one
            if self._tool != "eraser" and not self.has_active_label:
                log.warning("Drawing blocked: no active label")
                return

            if self._tool == "connect":
                self._connect_click(event.pos())
                self._emit_spectrum(event.pos(), force=True)

            elif self._tool == "circle":
                self._drawing = True
                self._circle_center = event.pos()
                self._circle_radius = 0
                self._circle_base_mask = self._mask.copy()
                self._draw_circle_preview(0)
                self._emit_spectrum(event.pos(), force=True)

            elif self._tool == "eraser":
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
        self._emit_spectrum(event.pos())
        # Do NOT emit signals.updated here — that would rebuild bboxes and
        # trigger label-spectra computation on every mouse-move event.

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

        if self._tool in ("fill", "connect"):
            return

        if self._drawing and (event.buttons() & Qt.LeftButton):
            self._draw_line(self._last_pos, event.pos())
            self._last_pos = event.pos()
            self._emit_spectrum(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._tool == "eraser":
                self._drawing = False
                self.signals.updated.emit()
                self.signals.shape_closed.emit()

            elif self._tool == "circle":
                if self._drawing:
                    self._paint_circle(self._circle_center, self._circle_radius)
                self._drawing = False
                self._circle_center = None
                self._circle_radius = 0
                self._circle_base_mask = None
                self.signals.updated.emit()
                # NOTE: label-spectra are computed on right-mouse release (see below)

        elif event.button() == Qt.RightButton and self._tool == "circle":
            # Circle is already committed on left-release; right-release is the
            # explicit trigger to compute label spectra (keeps drawing fluid).
            self.signals.shape_closed.emit()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _make_pen(self):
        if self._tool == "eraser":
            return QPen(Qt.transparent, self._pen_width * 3,
                        Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        color = QColor(self._pen_color)
        color.setAlpha(220)
        return QPen(color, self._pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

    def _paint_on_mask(self, painter_fn):
        """Paint directly onto the shared mask."""
        painter = QPainter(self._mask)
        if self._tool == "eraser":
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._tool != "circle":
            painter.setPen(self._make_pen())
        painter_fn(painter)
        painter.end()
        self.setPixmap(QPixmap.fromImage(self._mask))

    def recolor_label_pixels(self, old_color, new_color):
        if self._mask is None:
            return
        image = self._mask.convertToFormat(QImage.Format_RGBA8888)
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4)).copy()
        mask = _match_color(arr, QColor(*old_color, alpha=255))
        if not np.any(mask):
            return
        new_rgb = np.array(new_color, dtype=np.uint8)
        arr[mask, :3] = new_rgb
        self._mask = QImage(
            arr.data, width, height, width * 4, QImage.Format_RGBA8888
        ).copy().convertToFormat(QImage.Format_ARGB32)
        self.setPixmap(QPixmap.fromImage(self._mask))
        self.signals.updated.emit()

    def hide_label(self, label_id, color_tuple):
        """Temporarily hide all pixels of a label by storing their RGBA values
        and setting those pixels to transparent in the shared mask.
        """
        if self._mask is None:
            return
        image = self._mask.convertToFormat(QImage.Format_RGBA8888)
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4)).copy()
        hit = _match_color(arr, QColor(*color_tuple, alpha=255))
        n = int(np.count_nonzero(hit))
        if n == 0:
            log.debug("hide_label: no pixels found for label %s", label_id)
            return
        # store fragment
        frag = np.zeros_like(arr, dtype=np.uint8)
        frag[hit] = arr[hit]
        self._hidden_label_masks[label_id] = frag
        # clear pixels in main mask
        arr[hit] = 0
        self._mask = QImage(
            arr.data, width, height, width * 4, QImage.Format_RGBA8888
        ).copy().convertToFormat(QImage.Format_ARGB32)
        self.setPixmap(QPixmap.fromImage(self._mask))
        log.info("hide_label: hid %d pixels for label %s", n, label_id)
        self.signals.updated.emit()

    def show_label(self, label_id):
        """Restore previously hidden pixels for label_id (if any)."""
        if self._mask is None:
            return
        frag = self._hidden_label_masks.get(label_id)
        if frag is None:
            log.debug("show_label: no stored fragment for label %s", label_id)
            return
        image = self._mask.convertToFormat(QImage.Format_RGBA8888)
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4)).copy()
        mask_frag = frag[:, :, 3] > 0
        arr[mask_frag] = frag[mask_frag]
        self._mask = QImage(
            arr.data, width, height, width * 4, QImage.Format_RGBA8888
        ).copy().convertToFormat(QImage.Format_ARGB32)
        self.setPixmap(QPixmap.fromImage(self._mask))
        del self._hidden_label_masks[label_id]
        log.info("show_label: restored pixels for label %s", label_id)
        self.signals.updated.emit()

    def erase_label_pixels(self, color_tuple):
        """Erase all pixels matching color_tuple (R,G,B) from the mask (set to transparent)."""
        if self._mask is None:
            return
        image = self._mask.convertToFormat(QImage.Format_RGBA8888)
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4)).copy()
        hit = _match_color(arr, QColor(*color_tuple, alpha=255))
        n = int(np.count_nonzero(hit))
        if n == 0:
            log.debug("erase_label_pixels: no pixels found for color %s", color_tuple)
            return
        arr[hit] = 0  # fully transparent
        self._mask = QImage(
            arr.data, width, height, width * 4, QImage.Format_RGBA8888
        ).copy().convertToFormat(QImage.Format_ARGB32)
        self.setPixmap(QPixmap.fromImage(self._mask))
        log.info("erase_label_pixels: erased %d pixel(s) for color %s", n, color_tuple)
        self.signals.updated.emit()
        self.signals.shape_closed.emit()

    def _draw_dot(self, pos):
        self._paint_on_mask(lambda p: p.drawPoint(pos))

    def _draw_line(self, p1, p2):
        self._paint_on_mask(lambda p: p.drawLine(p1, p2))

    def _paint_circle(self, center, radius):
        if center is None or radius <= 0:
            return
        painter = QPainter(self._mask)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(self._pen_color)
        color.setAlpha(220)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(center, radius, radius)
        painter.end()
        self.setPixmap(QPixmap.fromImage(self._mask))
        self.signals.updated.emit()

    def _draw_circle_preview(self, radius):
        if self._circle_center is None or self._circle_base_mask is None:
            return
        preview_mask = self._circle_base_mask.copy()
        painter = QPainter(preview_mask)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(self._pen_color)
        color.setAlpha(220)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(self._circle_center, radius, radius)
        painter.end()
        self.setPixmap(QPixmap.fromImage(preview_mask))

    def _fill_connect_polygon(self):
        if len(self._connect_points) < 3:
            return
        painter = QPainter(self._mask)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(self._pen_color)
        color.setAlpha(220)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPolygon(QPolygonF(self._connect_points))
        painter.end()
        self.setPixmap(QPixmap.fromImage(self._mask))

    def _connect_click(self, pos):
        if self._connect_last is None and self._connect_points:
            self._connect_points = []
            self._connect_start = None
        point = QPointF(pos)
        if self._connect_last is None:
            # First point of a new path — mark as actively drawing
            self._connect_start = QPointF(point)
            self._connect_last = QPointF(point)
            self._connect_points = [QPointF(point)]
            self._drawing = True
            self._draw_dot(point)
            self.signals.updated.emit()
            return
        if len(self._connect_points) >= 3:
            dx = point.x() - self._connect_start.x()
            dy = point.y() - self._connect_start.y()
            if math.hypot(dx, dy) <= max(5.0, self._pen_width):
                # Auto-close: clicked near the start point
                self._draw_line(self._connect_last, self._connect_start)
                self._connect_points.append(QPointF(self._connect_start))
                self._fill_connect_polygon()
                self._connect_start = None
                self._connect_last = None
                self._connect_points = []
                self._drawing = False
                self.signals.updated.emit()
                self.signals.shape_closed.emit()
                log.debug("Connect path auto-closed (clicked near start)")
                return
        self._draw_line(self._connect_last, point)
        self._connect_last = QPointF(point)
        self._connect_points.append(QPointF(point))
        self.signals.updated.emit()

    def _close_connect_path(self):
        if self._connect_last is None or self._connect_start is None:
            self._connect_start = None
            self._connect_last = None
            self._connect_points = []
            self._drawing = False
            return
        if self._connect_last != self._connect_start:
            self._draw_line(self._connect_last, self._connect_start)
        n_pts = len(self._connect_points)
        if n_pts >= 3:
            self._fill_connect_polygon()
        self._connect_start = None
        self._connect_last = None
        self._connect_points = []
        self._drawing = False
        self.signals.updated.emit()
        self.signals.shape_closed.emit()
        log.debug("Connect path closed via right-click (%d points)", n_pts)

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
