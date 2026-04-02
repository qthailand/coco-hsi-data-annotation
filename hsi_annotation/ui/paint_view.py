from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QGraphicsView


class PaintView(QGraphicsView):
    ZI = 1.15
    ZO = 1 / 1.15
    ZMIN = 0.05
    ZMAX = 50.0

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self._zoom = 1.0
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setBackgroundBrush(QColor("#404040"))

    def zoom_in(self):
        self._apply(self.ZI)

    def zoom_out(self):
        self._apply(self.ZO)

    def zoom_reset(self):
        self.resetTransform()
        self._zoom = 1.0
        self._notify()

    def _apply(self, factor):
        new_zoom = self._zoom * factor
        if self.ZMIN <= new_zoom <= self.ZMAX:
            self._zoom = new_zoom
            self.scale(factor, factor)
            self._notify()

    def _notify(self):
        window = self.window()
        if hasattr(window, "update_zoom_label"):
            window.update_zoom_label(self._zoom)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            fake = event.__class__(
                event.type(), event.pos(), Qt.LeftButton, Qt.LeftButton, event.modifiers()
            )
            super().mousePressEvent(fake)
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.NoDrag)
            return
        super().mouseReleaseEvent(event)