import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path


def select_coco_json_file(prompt="Select COCO JSON annotations file"):
    # 1) ENV override, 2) cmdline arg, 3) PyQt5 file dialog
    env_path = os.environ.get("COCO_JSON_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        return sys.argv[1]

    try:
        from PyQt5.QtWidgets import QApplication, QFileDialog

        app = QApplication.instance()
        owns_app = app is None
        if owns_app:
            app = QApplication(sys.argv)

        path, _ = QFileDialog.getOpenFileName(
            None,
            prompt,
            "",
            "COCO JSON (*.json);;All files (*)",
        )

        if owns_app:
            app.quit()

        if path:
            return path
    except Exception as e:
        raise RuntimeError(f"PyQt5 file dialog failed: {e}")

    raise FileNotFoundError("COCO JSON file not found or not selected")


def load_coco_json(path):
    with open(path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    assert "images" in coco and len(
        coco["images"]) > 0, "Missing images in COCO file"
    assert "annotations" in coco, "Missing annotations in COCO file"
    return coco


def polygon_to_mask(segmentation, width, height):
    mask = np.zeros((height, width), dtype=bool)
    for poly in segmentation:
        if len(poly) < 6:
            continue
        coords = np.asarray(poly, dtype=np.float32).reshape(-1, 2)

        # crop bounding box to speed broad-phase
        x0, y0 = int(np.floor(coords[:, 0].min())), int(
            np.floor(coords[:, 1].min()))
        x1, y1 = int(np.ceil(coords[:, 0].max())), int(
            np.ceil(coords[:, 1].max()))
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(width - 1, x1)
        y1 = min(height - 1, y1)
        if x1 < x0 or y1 < y0:
            continue

        path = Path(coords)
        xx, yy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        points = np.vstack((xx.ravel(), yy.ravel())).T
        inside = path.contains_points(points)
        mask[y0: y1 + 1, x0: x1 + 1][inside.reshape(yy.shape)] = True

    return mask


def load_background_image(coco_json_path):
    folder = os.path.dirname(coco_json_path)
    base = os.path.splitext(os.path.basename(coco_json_path))[0]
    for ext in [".tif", ".tiff"]:
        candidate = os.path.join(folder, base + ext)
        if os.path.isfile(candidate):
            try:
                import imageio

                img = imageio.v3.imread(candidate)
                return img
            except Exception:
                try:
                    from PIL import Image

                    return np.asarray(Image.open(candidate).convert("RGB"))
                except Exception:
                    pass
    return None


def render_coco_annotations(coco, coco_json_path=None, show=True, save_prefix=None):
    image = coco["images"][0]
    width = int(image.get("width", 0))
    height = int(image.get("height", 0))

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions are invalid in COCO JSON")

    background = None
    if coco_json_path:
        background = load_background_image(coco_json_path)
        if background is not None:
            if background.ndim == 3 and background.shape[2] == 4:
                background = background[:, :, :3]

    category_name_map = {}
    for cat in coco.get("categories", []):
        cid = cat.get("id")
        cname = cat.get("name", f"class_{cid}")
        if cid is not None:
            category_name_map[int(cid)] = cname

    for ann in coco["annotations"]:
        annotation_id = ann.get("id")
        category_id = ann.get("category_id")
        class_name = category_name_map.get(category_id, f"class_{category_id}")
        bbox = ann.get("bbox")
        segmentation = ann.get("segmentation", [])

        mask = np.zeros((height, width), dtype=bool)
        if isinstance(segmentation, list) and len(segmentation) > 0:
            if isinstance(segmentation[0], list):
                mask = polygon_to_mask(segmentation, width, height)

        fig, ax = plt.subplots(figsize=(6, 6))

        if background is not None:
            ax.imshow(background)
        else:
            base = np.zeros((height, width, 3), dtype=np.uint8) + 32
            ax.imshow(base)

        ax.imshow(np.ma.masked_where(~mask, mask), cmap="Reds", alpha=0.6)

        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            rect = plt.Rectangle((x, y), w, h, fill=False,
                                 edgecolor="cyan", linewidth=2)
            ax.add_patch(rect)

            # draw class name in bbox
            label = f"{class_name} (sample id={annotation_id})"
            ax.text(
                x,
                max(y - 2, 0),
                label,
                fontsize=8,
                color="white",
                verticalalignment="bottom",
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 2},
            )

        title = f"ann_id={annotation_id} cat={
            category_id} bbox={bbox} name={class_name}"
        ax.set_title(title)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")

        if save_prefix:
            save_path = f"{save_prefix}_ann_{annotation_id}.png"
            fig.savefig(save_path, bbox_inches="tight")
            print(f"saved: {save_path}")

        if show:
            plt.show()

        plt.close(fig)


def test_coco_annotation_rendering(path=None):
    if path is None:
        path = select_coco_json_file()

    coco = load_coco_json(path)
    assert len(coco.get("annotations", [])) > 0, "COCO JSON has no annotations"

    # sanity check image metadata
    image = coco["images"][0]
    assert int(image["width"]) > 0
    assert int(image["height"]) > 0

    # each annotation must have bbox seg
    for ann in coco["annotations"]:
        assert "id" in ann
        assert "category_id" in ann
        assert "bbox" in ann and len(ann["bbox"]) == 4

    # plot each annotation one by one with the background for visual testing
    render_coco_annotations(coco, coco_json_path=path,
                            show=True, save_prefix="coco_ann")


if __name__ == "__main__":
    test_coco_annotation_rendering()
