import re

import numpy as np
import spectral as spy
from PyQt5.QtGui import QImage


RGB_BANDS = (29, 19, 9)
RGB_TARGET_WAVELENGTHS_VIS = (645.0, 555.0, 465.0)
RGB_TARGET_WAVELENGTHS_SWIR = (1500.0, 1300.0, 1100.0)
RGB_TARGET_WAVELENGTHS = RGB_TARGET_WAVELENGTHS_VIS
COLOR_TOLERANCE = 3
DEFAULT_LOW_CUT = 2.0
DEFAULT_HIGH_CUT = 98.0


def load_datacube_preview(
    path,
    low_cut=DEFAULT_LOW_CUT,
    high_cut=DEFAULT_HIGH_CUT,
    target_wavelengths=None,
):
    datacube = spy.open_image(path)
    rgb, preview_info = build_rgb_preview(
        datacube,
        low_cut=low_cut,
        high_cut=high_cut,
        target_wavelengths=target_wavelengths,
    )
    rgb8 = np.ascontiguousarray((rgb * 255).clip(0, 255).astype(np.uint8))
    height, width, _ = rgb8.shape
    qimg = QImage(rgb8.data, width, height, width * 3, QImage.Format_RGB888)
    return datacube, qimg.convertToFormat(QImage.Format_ARGB32).copy(), preview_info


def _select_default_target_wavelengths(metadata):
    wavelengths = extract_wavelengths(metadata)
    if wavelengths is None or len(wavelengths) == 0:
        return RGB_TARGET_WAVELENGTHS_VIS
    if np.nanmean(wavelengths) > 1000.0 or np.nanmin(wavelengths) > 900.0:
        return RGB_TARGET_WAVELENGTHS_SWIR
    return RGB_TARGET_WAVELENGTHS_VIS


def build_rgb_preview(
    datacube,
    low_cut=DEFAULT_LOW_CUT,
    high_cut=DEFAULT_HIGH_CUT,
    target_wavelengths=None,
):
    if target_wavelengths is None:
        target_wavelengths = _select_default_target_wavelengths(datacube.metadata)
    band_indices, actual_wavelengths = select_rgb_bands(datacube, target_wavelengths)
    rgb = np.asarray(datacube.read_bands(list(band_indices)), dtype=np.float32)
    rgb = _percentile_stretch_rgb(rgb, low_cut, high_cut)
    preview_info = {
        "band_indices": tuple(int(index) for index in band_indices),
        "target_wavelengths": tuple(float(value) for value in target_wavelengths),
        "actual_wavelengths": actual_wavelengths,
        "low_cut": float(low_cut),
        "high_cut": float(high_cut),
        "used_metadata_wavelengths": actual_wavelengths is not None,
    }
    return rgb, preview_info


def select_rgb_bands(datacube, target_wavelengths=None):
    if target_wavelengths is None:
        target_wavelengths = _select_default_target_wavelengths(datacube.metadata)
    wavelengths = extract_wavelengths(datacube.metadata)
    if wavelengths is None or len(wavelengths) == 0:
        return RGB_BANDS, None
    band_indices = []
    actual_wavelengths = []
    for target in target_wavelengths:
        index = int(np.argmin(np.abs(wavelengths - target)))
        band_indices.append(index)
        actual_wavelengths.append(float(wavelengths[index]))
    return tuple(band_indices), tuple(actual_wavelengths)


def extract_wavelengths(metadata):
    raw_wavelengths = metadata.get("wavelength") if metadata else None
    if raw_wavelengths is None:
        return None
    values = _coerce_wavelength_values(raw_wavelengths)
    if not values:
        return None
    wavelengths = np.asarray(values, dtype=np.float32)
    units = str(metadata.get("wavelength units", "")).strip().lower() if metadata else ""
    if _uses_micrometer_units(units, wavelengths):
        wavelengths = wavelengths * 1000.0
    return wavelengths


def compute_class_spectra(datacube, mask, classes, max_samples=300, progress_callback=None):
    if progress_callback is not None:
        progress_callback(0.0)
    if datacube is None:
        if progress_callback is not None:
            progress_callback(1.0)
        return [(name, color, None) for _, name, color in classes]
    mask_arr = _qimage_to_rgba_array(mask)
    class_data = []
    for _, name, color in classes:
        match = _match_color(mask_arr, color)
        ys, xs = np.where(match)
        if len(ys) == 0:
            class_data.append((name, color, None))
        else:
            if len(ys) > max_samples:
                idx = np.random.choice(len(ys), max_samples, replace=False)
                ys, xs = ys[idx], xs[idx]
            valid = (xs < datacube.ncols) & (ys < datacube.nrows)
            ys, xs = ys[valid], xs[valid]
            if len(ys) == 0:
                class_data.append((name, color, None))
            else:
                try:
                    spectra = np.array(
                        [
                            np.array(datacube[int(y), int(x), :], dtype=np.float32).flatten()
                            for y, x in zip(ys, xs)
                        ]
                    )
                    class_data.append((name, color, spectra.mean(axis=0)))
                except Exception:
                    class_data.append((name, color, None))
    if progress_callback is not None:
        progress_callback(1.0)
    return class_data


def build_class_id_mask(mask, classes):
    mask_arr = _qimage_to_rgba_array(mask)
    height, width = mask_arr.shape[:2]
    id_arr = np.zeros((height, width), dtype=np.uint8)
    for class_id, _, color in classes:
        id_arr[_match_color(mask_arr, color)] = class_id
    return np.ascontiguousarray(id_arr)


def _iter_connected_components(class_id_mask):
    height, width = class_id_mask.shape
    visited = np.zeros_like(class_id_mask, dtype=bool)
    for y in range(height):
        for x in range(width):
            class_id = int(class_id_mask[y, x])
            if class_id == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cx, cy))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                        if class_id_mask[ny, nx] == class_id:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            yield class_id, pixels


def build_coco_annotations_from_mask(class_id_mask, image_id=1, segmentation_method="polygon"):
    annotations = []
    annotation_id = 1
    for class_id, pixels in _iter_connected_components(class_id_mask):
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        x_min = int(min(xs))
        x_max = int(max(xs))
        y_min = int(min(ys))
        y_max = int(max(ys))
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        area = len(pixels)
        if w <= 0 or h <= 0 or area <= 0:
            continue
        if segmentation_method == "polygon":
            binary = (class_id_mask == class_id).astype(np.uint8)
            segmentation = _mask_to_polygons(binary)
            if not segmentation:
                # fallback to bbox rectangle if contour extraction fails
                segmentation = [[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]]
        else:
            segmentation = [[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]]
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "segmentation": segmentation,
                "area": area,
                "bbox": [x_min, y_min, w, h],
                "iscrowd": 0,
            }
        )
        annotation_id += 1
    return annotations


def build_coco_annotation_json(mask, classes, image_id=1, file_name=""):
    class_id_mask = build_class_id_mask(mask, classes)
    height, width = class_id_mask.shape
    categories = [
        {"id": int(class_id), "name": str(name), "supercategory": ""}
        for class_id, name, _ in classes
    ]
    coco = {
        "info": {
            "description": "HSI annotation converted to COCO format",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [],
        "images": [{"id": image_id, "file_name": file_name or "", "height": height, "width": width}],
        "annotations": build_coco_annotations_from_mask(class_id_mask, image_id=image_id),
        "categories": categories,
    }
    return coco


def _mask_to_polygons(binary_mask, min_points=6):
    """
    Convert a binary uint8 mask to a list of COCO segmentation polygons.
    Each polygon is a flat list [x1,y1, x2,y2, ...] with at least min_points.
    Uses cv2.findContours with RETR_EXTERNAL so holes are ignored and only
    outer contours are returned.  Each connected region becomes one polygon.
    Returns [] if cv2 is unavailable or no valid contour is found.
    """
    try:
        import cv2
    except ImportError:
        return []

    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    polygons = []
    for contour in contours:
        # contour shape: (N, 1, 2)
        contour = contour.squeeze(axis=1)   # → (N, 2)
        if len(contour) < 3:
            continue
        flat = contour.flatten().tolist()   # [x1,y1, x2,y2, ...]
        if len(flat) < min_points:
            continue
        polygons.append(flat)
    return polygons


def _qimage_to_rgba_array(image):
    rgba = image.convertToFormat(QImage.Format_RGBA8888)
    width, height = rgba.width(), rgba.height()
    ptr = rgba.bits()
    ptr.setsize(height * width * 4)
    return np.frombuffer(ptr, np.uint8).reshape((height, width, 4)).copy()


def build_coco_annotations_from_layers(layers, image_id=1):
    annotations = []
    annotation_id = 1  # FIX: must be globally unique across all annotations

    for layer in layers:
        if not layer.get("visible", True):
            continue

        mask = layer.get("mask")
        if mask is None:
            continue

        arr = _qimage_to_rgba_array(mask)
        alpha = arr[:, :, 3]
        ys, xs = np.where(alpha > 0)

        if xs.size == 0 or ys.size == 0:
            continue

        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        area = int((alpha > 0).sum())

        # Pixel-level segmentation polygons from alpha mask contours
        segmentation = _mask_to_polygons(alpha)
        if not segmentation:
            # fallback to bbox rectangle if cv2 unavailable or contour extraction fails
            segmentation = [[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]]

        class_ids = layer.get("class_ids")
        if class_ids is None:
            class_ids = [layer.get("class_id", 1)]

        for cid in class_ids:
            annotations.append(
                {
                    "id": annotation_id,  # FIX: was layer["id"], causing duplicates when class_ids > 1
                    "image_id": int(image_id),
                    "category_id": int(cid),
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": [x_min, y_min, w, h],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return annotations


def build_coco_annotation_json_from_layers(layers, classes=None, image_id=1, file_name=""):
    annotations = build_coco_annotations_from_layers(layers, image_id=image_id)

    if classes is None:
        category_ids = sorted({ann["category_id"] for ann in annotations})
        categories = [
            {"id": cid, "name": f"class_{cid}", "supercategory": ""}
            for cid in category_ids
        ]
    else:
        class_map = (
            {c[0]: c[1] for c in classes}
            if isinstance(classes, list)
            else {k: v["name"] for k, v in classes.items()}
        )
        category_ids = sorted({ann["category_id"] for ann in annotations})
        categories = [
            {"id": cid, "name": class_map.get(cid, f"class_{cid}"), "supercategory": ""}
            for cid in category_ids
        ]

    # FIX: find image size from first layer that actually has a valid mask
    image_width, image_height = 0, 0
    for layer in layers:
        m = layer.get("mask")
        if m is not None:
            image_width, image_height = m.width(), m.height()
            break

    coco = {
        "info": {
            "description": "COCO 2017 Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2017,
            "contributor": "COCO Consortium",
            "date_created": "2017/09/01",
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by/2.0/",
                "id": 4,
                "name": "Attribution License",
            }
        ],
        "images": [
            {
                "id": image_id,
                "license": 4,
                "width": image_width,
                "height": image_height,
                "file_name": file_name or "",
                "date_captured": "2013-11-15 02:41:42",
            }
        ],
        "annotations": annotations,
        "categories": categories,
    }

    return coco


def _match_color(mask_arr, color, tolerance=COLOR_TOLERANCE):
    return (
        (np.abs(mask_arr[:, :, 0].astype(int) - color.red()) <= tolerance)
        & (np.abs(mask_arr[:, :, 1].astype(int) - color.green()) <= tolerance)
        & (np.abs(mask_arr[:, :, 2].astype(int) - color.blue()) <= tolerance)
        & (mask_arr[:, :, 3] > 0)
    )


def _coerce_wavelength_values(raw_wavelengths):
    if isinstance(raw_wavelengths, str):
        cleaned = raw_wavelengths.strip().strip("{}").strip("[]")
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    else:
        parts = list(raw_wavelengths)
    values = []
    for part in parts:
        if isinstance(part, (float, int)):
            values.append(float(part))
            continue
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(part))
        if match is not None:
            values.append(float(match.group(0)))
    return values


def _uses_micrometer_units(units, wavelengths):
    if "mic" in units or "um" in units:
        return True
    if "nm" in units:
        return False
    return float(np.nanmedian(wavelengths)) < 20.0


def _percentile_stretch_rgb(rgb, low_cut, high_cut):
    if not 0.0 <= low_cut < high_cut <= 100.0:
        raise ValueError("Contrast cuts must satisfy 0 <= low_cut < high_cut <= 100")
    stretched = np.empty_like(rgb, dtype=np.float32)
    for channel_index in range(rgb.shape[2]):
        channel = rgb[:, :, channel_index].astype(np.float32, copy=False)
        low = float(np.percentile(channel, low_cut))
        high = float(np.percentile(channel, high_cut))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(channel))
            high = float(np.max(channel))
        if high <= low:
            stretched[:, :, channel_index] = np.zeros_like(channel, dtype=np.float32)
            continue
        stretched[:, :, channel_index] = np.clip((channel - low) / (high - low), 0.0, 1.0)
    return stretched
