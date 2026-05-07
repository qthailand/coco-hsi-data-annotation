"""
registry.py
-----------
Central data model for labels and annotations.

LabelRegistry  — dict store  {label_id: {"name": str, "color": (R,G,B), "visible": bool}}
AnnotationRegistry — list store [{"id": int, "label_id": int, "bbox": [...], "segmentation": [...]}]

Both emit Qt signals on every mutation so the UI stays in sync.
"""

import logging

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor

log = logging.getLogger(__name__)


_DEFAULT_COLORS = [
    (231,  76,  60),
    ( 46, 204, 113),
    ( 52, 152, 219),
    (241, 196,  15),
    (155,  89, 182),
    ( 26, 188, 156),
    (230, 126,  34),
    ( 52,  73,  94),
    (236, 112, 176),
    (149, 165, 166),
]


class LabelRegistry(QObject):
    """
    Signals: label_added(id), label_about_to_be_removed(id, color_tuple),
             label_removed(id), label_changed(id),
             label_color_changed(id, old_color, new_color), reset()
    """
    label_added                = pyqtSignal(int)
    label_about_to_be_removed  = pyqtSignal(int, object)  # (id, (R,G,B)) — fires BEFORE deletion
    label_removed              = pyqtSignal(int)
    label_changed              = pyqtSignal(int)
    label_color_changed        = pyqtSignal(int, object, object)
    reset                      = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._labels = {}       # {int: {"name": str, "color": (R,G,B), "visible": bool}}
        self._next_id = 1

    # --- read ---

    def ids(self):
        return sorted(self._labels.keys())

    def get(self, label_id):
        return self._labels.get(label_id)

    def name(self, label_id):
        e = self._labels.get(label_id)
        return e["name"] if e else ""

    def color(self, label_id):
        """Return (R, G, B) tuple."""
        e = self._labels.get(label_id)
        return e["color"] if e else (128, 128, 128)

    def qcolor(self, label_id, alpha=220):
        r, g, b = self.color(label_id)
        return QColor(r, g, b, alpha)

    def is_visible(self, label_id):
        e = self._labels.get(label_id)
        return e["visible"] if e else True

    def __len__(self):
        return len(self._labels)

    def __contains__(self, label_id):
        return label_id in self._labels

    def as_list(self):
        """[(id, name, QColor), ...] sorted by id — for canvas/data helpers."""
        return [(lid, self._labels[lid]["name"], self.qcolor(lid)) for lid in self.ids()]

    # --- mutations ---

    def add_label(self, name=None, color=None, label_id=None):
        """Add a new label; returns assigned label_id."""
        if label_id is not None:
            if label_id in self._labels:
                raise ValueError("label_id {} already exists".format(label_id))
            lid = label_id
            self._next_id = max(self._next_id, lid + 1)
        else:
            lid = self._next_id
            self._next_id += 1

        if name is None:
            name = "Label {}".format(lid)
        if color is None:
            color = _DEFAULT_COLORS[(lid - 1) % len(_DEFAULT_COLORS)]

        self._labels[lid] = {"name": name, "color": tuple(color), "visible": True}
        self.label_added.emit(lid)
        return lid

    def remove_label(self, label_id):
        if label_id not in self._labels:
            return False
        old_color = tuple(self._labels[label_id]["color"])
        old_name  = self._labels[label_id]["name"]
        log.info("Removing label id=%d  name='%s'  color=%s", label_id, old_name, old_color)
        # Fire BEFORE deletion so subscribers can still use the color
        self.label_about_to_be_removed.emit(label_id, old_color)
        del self._labels[label_id]
        self.label_removed.emit(label_id)
        return True

    def set_name(self, label_id, name):
        if label_id not in self._labels:
            return False
        self._labels[label_id]["name"] = str(name)
        self.label_changed.emit(label_id)
        return True

    def set_color(self, label_id, color):
        """color: (R,G,B) tuple or QColor."""
        if label_id not in self._labels:
            return False
        if isinstance(color, QColor):
            color = (color.red(), color.green(), color.blue())
        old_color = tuple(self._labels[label_id]["color"])
        new_color = tuple(int(v) for v in color)
        self._labels[label_id]["color"] = new_color
        self.label_changed.emit(label_id)
        if old_color != new_color:
            self.label_color_changed.emit(label_id, old_color, new_color)
        return True

    def set_visible(self, label_id, visible):
        if label_id not in self._labels:
            return False
        self._labels[label_id]["visible"] = bool(visible)
        self.label_changed.emit(label_id)
        return True

    def clear(self):
        self._labels.clear()
        self._next_id = 1
        self.reset.emit()

    def load(self, data):
        """Bulk-load from {id: entry} dict. Emits reset()."""
        self._labels.clear()
        self._next_id = 1
        for lid, entry in data.items():
            lid = int(lid)
            self._labels[lid] = {
                "name":    str(entry.get("name", "Label {}".format(lid))),
                "color":   tuple(int(v) for v in entry.get("color", (128, 128, 128))),
                "visible": bool(entry.get("visible", True)),
            }
            self._next_id = max(self._next_id, lid + 1)
        self.reset.emit()

    def to_dict(self):
        return {lid: dict(e) for lid, e in self._labels.items()}


class AnnotationRegistry(QObject):
    """
    Signals: annotation_added(id), annotation_removed(id), annotation_changed(id), reset()
    """
    annotation_added   = pyqtSignal(int)
    annotation_removed = pyqtSignal(int)
    annotation_changed = pyqtSignal(int)
    reset              = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._annotations = []  # [{"id", "label_id", "bbox", "segmentation", "area"}]
        self._next_id = 1

    def __len__(self):
        return len(self._annotations)

    def __iter__(self):
        return iter(self._annotations)

    def get(self, annotation_id):
        for a in self._annotations:
            if a["id"] == annotation_id:
                return a
        return None

    def by_label(self, label_id):
        return [a for a in self._annotations if a["label_id"] == label_id]

    def all(self):
        return list(self._annotations)

    def add(self, label_id, bbox=None, segmentation=None, area=0):
        ann_id = self._next_id
        self._next_id += 1
        self._annotations.append({
            "id":           ann_id,
            "label_id":     label_id,
            "bbox":         list(bbox or []),
            "segmentation": list(segmentation or []),
            "area":         area,
        })
        self.annotation_added.emit(ann_id)
        return ann_id

    def remove(self, annotation_id):
        for i, a in enumerate(self._annotations):
            if a["id"] == annotation_id:
                self._annotations.pop(i)
                self.annotation_removed.emit(annotation_id)
                return True
        return False

    def remove_by_label(self, label_id):
        for ann_id in [a["id"] for a in self._annotations if a["label_id"] == label_id]:
            self.remove(ann_id)

    def update(self, annotation_id, **kwargs):
        a = self.get(annotation_id)
        if a is None:
            return False
        for k, v in kwargs.items():
            if k != "id":
                a[k] = v
        self.annotation_changed.emit(annotation_id)
        return True

    def clear(self):
        self._annotations.clear()
        self._next_id = 1
        self.reset.emit()

    def load(self, data):
        self._annotations.clear()
        self._next_id = 1
        for entry in data:
            ann = {
                "id":           int(entry.get("id", self._next_id)),
                "label_id":     int(entry["label_id"]),
                "bbox":         list(entry.get("bbox", [])),
                "segmentation": list(entry.get("segmentation", [])),
                "area":         int(entry.get("area", 0)),
            }
            self._annotations.append(ann)
            self._next_id = max(self._next_id, ann["id"] + 1)
        self.reset.emit()

    def to_list(self):
        return [dict(a) for a in self._annotations]
