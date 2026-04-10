"""Ultralytics segmentation masks to normalized polygon coordinates (noise_deletion_clean YOLO label style)."""

from __future__ import annotations

import os
import sys
from typing import Any

_third = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party"))
if _third not in sys.path:
    sys.path.insert(0, _third)


def result_to_polygons_by_class(
    result: Any,
    class_name_to_yolo_id: dict[str, int],
    img_shape: tuple[int, int, int],
) -> dict[int, list[list[tuple[float, float]]]]:
    h, w = img_shape[0], img_shape[1]
    wx = max(w - 1, 1)
    hx = max(h - 1, 1)
    out: dict[int, list[list[tuple[float, float]]]] = {}
    if result.masks is None:
        return out
    for i, mask_xy in enumerate(result.masks.xy):
        cls_idx = int(result.boxes.cls[i])
        name = result.names[cls_idx]
        if name not in class_name_to_yolo_id:
            continue
        cid = class_name_to_yolo_id[name]
        poly = [(float(px) / wx, float(py) / hx) for px, py in mask_xy]
        if len(poly) >= 3:
            out.setdefault(cid, []).append(poly)
    return out


def write_yolo_seg_label_txt(path: str, polygons_by_class: dict[int, list[list[tuple[float, float]]]]) -> None:
    lines = []
    for cid in sorted(polygons_by_class.keys()):
        for poly in polygons_by_class[cid]:
            if len(poly) < 3:
                continue
            coords = [f"{x} {y}" for x, y in poly]
            lines.append(f"{cid} " + " ".join(coords))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def write_merged_label_with_noise(
    path: str,
    polygons_by_class: dict[int, list[list[tuple[float, float]]]],
    noise_mask,
    height: int,
    width: int,
) -> None:
    from noise_deletion_clean.masks import mask_to_normalized_polygons

    noise_polys = mask_to_normalized_polygons(noise_mask, height, width)
    merged = {
        0: list(polygons_by_class.get(0, [])),
        1: list(polygons_by_class.get(1, [])),
        2: noise_polys,
    }
    write_yolo_seg_label_txt(path, merged)
