"""
Dual-model stem pipeline: model A (Casparian + epidermis), model B (VB),
ring and casp noise stages, stem metrics, and visualization.
"""

from __future__ import annotations

import logging
import os
import sys

_third = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party"))
if _third not in sys.path:
    sys.path.insert(0, _third)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon
from shapely.validation import make_valid

from noise_deletion_clean.detection import detect_noise_mask
from noise_deletion_clean.masks import build_casp_inside_mask, build_ring_roi

from config.inference_constants import (
    CLASS_CASPARIAN,
    CLASS_CASPARIAN_FALLBACK,
    CLASS_EPIDERMIS,
    CLASS_VASCULAR_BUNDLES,
    MODEL_A_EPIDERMIS_KEEP_NEAREST_TO_CENTER,
    MODEL_A_SWAP_EPI_CASP_LABELS,
    PIXEL_TO_MICRON,
    YOLO_CLASS_CASPARIAN,
    YOLO_CLASS_EPIDERMIS,
)
from config.noise_profiles_app import DEFAULT_STAGE, app_params_for_stage
from stem_metrics import compute_stem_metric_row
from yolo_label_export import (
    result_to_polygons_by_class,
    write_merged_label_with_noise,
    write_yolo_seg_label_txt,
)

_post_logger = logging.getLogger("post_processor_debug")


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _class_map_model_a(result) -> dict[str, int]:
    """Map model class names to internal YOLO ids 0=casparian 1=epidermis (see inference_constants)."""
    names = set(result.names.values())
    m: dict[str, int] = {}
    if MODEL_A_SWAP_EPI_CASP_LABELS:
        if CLASS_EPIDERMIS in names:
            m[CLASS_EPIDERMIS] = YOLO_CLASS_CASPARIAN
        if CLASS_CASPARIAN in names:
            m[CLASS_CASPARIAN] = YOLO_CLASS_EPIDERMIS
        elif CLASS_CASPARIAN_FALLBACK in names:
            m[CLASS_CASPARIAN_FALLBACK] = YOLO_CLASS_CASPARIAN
    else:
        if CLASS_EPIDERMIS in names:
            m[CLASS_EPIDERMIS] = YOLO_CLASS_EPIDERMIS
        if CLASS_CASPARIAN in names:
            m[CLASS_CASPARIAN] = YOLO_CLASS_CASPARIAN
        elif CLASS_CASPARIAN_FALLBACK in names:
            m[CLASS_CASPARIAN_FALLBACK] = YOLO_CLASS_CASPARIAN
    return m


def _collect_polygons(result):
    preds = []
    if result.masks is None:
        return preds
    for i, mask_polygon in enumerate(result.masks.xy):
        cls = int(result.boxes.cls[i])
        bbox = result.boxes[i]
        poly = Polygon(mask_polygon)
        preds.append({
            "points": mask_polygon,
            "class_name": result.names[cls],
            "area": float(poly.area),
            "box": bbox.xyxy[0].cpu().numpy(),
        })
    return preds


def _polygon_center(points):
    return Polygon(points).centroid.x, Polygon(points).centroid.y


def _find_main_casparian(preds, img_shape):
    if MODEL_A_SWAP_EPI_CASP_LABELS:
        # Biology: casparian inside epidermis; model names that inner mask "Epidermis".
        casp_names = (CLASS_EPIDERMIS, CLASS_CASPARIAN_FALLBACK)
    else:
        casp_names = (CLASS_CASPARIAN, CLASS_CASPARIAN_FALLBACK)
    h, w = img_shape[:2]
    img_center = np.array([w / 2, h / 2])
    min_dist = float("inf")
    main_poly = None
    main_box = None
    for p in preds:
        if p["class_name"] not in casp_names:
            continue
        cx, cy = _polygon_center(p["points"])
        dist = np.linalg.norm(img_center - np.array([cx, cy]))
        if dist < min_dist:
            min_dist = dist
            main_poly = Polygon(p["points"])
            main_box = p
    return main_poly, main_box


def _filter_epidermis_polygons_nearest_to_center(
    norm_polys: list[list[tuple[float, float]]],
    h: int,
    w: int,
) -> list[list[tuple[float, float]]]:
    """Keep a single epidermis polygon: centroid closest to image center; tie-break by larger area."""
    if len(norm_polys) <= 1:
        return norm_polys
    img_center = np.array([w / 2.0, h / 2.0])
    wx = max(w - 1, 1)
    hx = max(h - 1, 1)
    scored: list[tuple[float, float, list[tuple[float, float]]]] = []
    for pts in norm_polys:
        if len(pts) < 3:
            continue
        try:
            pts_px = [(float(x) * wx, float(y) * hx) for x, y in pts]
            poly = Polygon(pts_px)
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.is_empty:
                continue
            if poly.geom_type == "Polygon":
                geoms = [poly]
            else:
                geoms = [g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon" and not g.is_empty]
            if not geoms:
                continue
            best_g = max(geoms, key=lambda g: g.area)
            c = best_g.centroid
            dist = float(np.linalg.norm(img_center - np.array([c.x, c.y])))
            scored.append((dist, -float(best_g.area), pts))
        except Exception:
            continue
    if not scored:
        return []
    scored.sort(key=lambda t: (t[0], t[1]))
    return [scored[0][2]]


def _filter_vascular_bundles(main_box, preds):
    if main_box is None:
        return []
    x1, y1, x2, y2 = main_box["box"][0], main_box["box"][1], main_box["box"][2], main_box["box"][3]
    vb_polys = []
    for p in preds:
        if p["class_name"] != CLASS_VASCULAR_BUNDLES:
            continue
        cx, cy = _polygon_center(p["points"])
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            vb_polys.append(p["points"])
    return vb_polys


def _draw_dual_viz(
    img_bgr,
    main_poly,
    vb_polys,
    noise_union,
    output_path,
    epidermis_polys: list[list[tuple[float, float]]] | None = None,
):
    """Overlay: noise tint, epidermis outline (red), main Casparian (deepskyblue), VB fills (green)."""
    h0, w0 = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.imshow(rgb)

    if noise_union is not None and noise_union.any():
        nu = (noise_union > 0).astype(np.uint8)
        ov = np.zeros((h0, w0, 4), dtype=np.float32)
        ov[..., 0] = 1.0
        ov[..., 3] = nu.astype(np.float32) * 0.35
        ax.imshow(ov)

    wx0 = max(w0 - 1, 1)
    hx0 = max(h0 - 1, 1)
    if epidermis_polys:
        for pts in epidermis_polys:
            try:
                pts_px = [(float(x) * wx0, float(y) * hx0) for x, y in pts]
                poly = Polygon(pts_px)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_empty:
                    continue
                if poly.geom_type == "Polygon":
                    geoms = [poly]
                else:
                    geoms = getattr(poly, "geoms", [poly])
                for g in geoms:
                    if g.is_empty or g.geom_type != "Polygon":
                        continue
                    x, y = g.exterior.xy
                    ax.plot(x, y, color="red", linewidth=1.5)
            except Exception:
                continue
    if main_poly is not None:
        x, y = main_poly.exterior.xy
        ax.plot(x, y, color="deepskyblue", linewidth=1.5)
    for vb in vb_polys:
        vp = Polygon(vb)
        x, y = vp.exterior.xy
        ax.fill(x, y, color="green", alpha=0.45)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


class DualStemPipelineProcessor:
    def __init__(self, output_dir: str, debug_log_dir: str | None = None, pixel_to_micron: float | None = None):
        self.viz_output_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.viz_output_dir, exist_ok=True)
        self.PIXEL_TO_MICRON = pixel_to_micron if pixel_to_micron is not None else PIXEL_TO_MICRON
        self.visualization_paths: list[str] = []
        log_dir = debug_log_dir if debug_log_dir else output_dir
        self._log_path = os.path.join(log_dir, "post_processor_debug.log")
        if debug_log_dir:
            os.makedirs(debug_log_dir, exist_ok=True)
        self._labels_dir = os.path.join(output_dir, "labels_generated")
        self._geom_dir = os.path.join(output_dir, "geometry_export")
        os.makedirs(self._labels_dir, exist_ok=True)
        os.makedirs(self._geom_dir, exist_ok=True)

    def _na_row(self, img_name: str, note: str) -> dict:
        return {
            "image_name": img_name,
            "ring_area_um2": "N/A",
            "ring_area_minus_noise_um2": "N/A",
            "mean_ring_thickness_um": "N/A",
            "mean_ring_thickness_minus_noise_um": "N/A",
            "casp_area_um2": "N/A",
            "casp_area_minus_noise_um2": "N/A",
            "epidermis_area_um2": "N/A",
            "epidermis_area_minus_noise_um2": "N/A",
            "vb_count": "N/A",
            "vb_area_microns": "N/A",
            "avg_vb_area_microns": "N/A",
            "cs_area_microns": "N/A",
            "vb_to_cs_ratio": "N/A",
            "notes": note,
        }

    def process_batch(
        self,
        image_paths: list[str],
        results_a: list,
        results_b: list,
        append_debug_log: bool = False,
    ) -> pd.DataFrame:
        records = []
        self.visualization_paths = []

        _post_logger.setLevel(logging.DEBUG)
        _post_logger.handlers.clear()
        log_mode = "a" if append_debug_log else "w"
        fh = logging.FileHandler(self._log_path, mode=log_mode, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        _post_logger.addHandler(fh)
        if append_debug_log and image_paths:
            _post_logger.info("--- %s ---", os.path.basename(image_paths[0]))
        else:
            _post_logger.info("=== Dual stem pipeline started ===\n")

        for img_path, ra, rb in zip(image_paths, results_a, results_b):
            img_name = os.path.basename(img_path)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                _post_logger.warning("Cannot read image %s", img_name)
                continue

            preds_a = _collect_polygons(ra)
            preds_b = _collect_polygons(rb)

            main_poly, main_box = _find_main_casparian(preds_a, img_bgr.shape)
            vb_polys = _filter_vascular_bundles(main_box, preds_b)

            if main_poly is None or main_box is None:
                _post_logger.warning("No main Casparian/Cross Section in %s", img_name)
                records.append(self._na_row(img_name, "No Casparian / Cross Section"))
                continue

            cmap = _class_map_model_a(ra)
            if not cmap:
                _post_logger.warning("Model A has no expected class names in %s", img_name)
                records.append(self._na_row(img_name, "Model A class names mismatch"))
                continue

            polys_by_class = result_to_polygons_by_class(ra, cmap, img_bgr.shape)
            if YOLO_CLASS_CASPARIAN not in polys_by_class or not polys_by_class.get(YOLO_CLASS_CASPARIAN):
                _post_logger.warning("No Casparian polygons after export in %s", img_name)
                records.append(self._na_row(img_name, "No Casparian mask"))
                continue
            if YOLO_CLASS_EPIDERMIS not in polys_by_class or not polys_by_class.get(YOLO_CLASS_EPIDERMIS):
                _post_logger.warning("No Epidermis polygons in %s (ring ROI needs class 1)", img_name)
                records.append(self._na_row(img_name, "No Epidermis mask"))
                continue

            h, w = img_bgr.shape[:2]
            if MODEL_A_EPIDERMIS_KEEP_NEAREST_TO_CENTER:
                epi_all = polys_by_class[YOLO_CLASS_EPIDERMIS]
                epi_one = _filter_epidermis_polygons_nearest_to_center(epi_all, h, w)
                if not epi_one:
                    _post_logger.warning("Epidermis nearest-to-center filter left no polygons in %s", img_name)
                    records.append(self._na_row(img_name, "Epidermis filter empty"))
                    continue
                if len(epi_all) > 1:
                    _post_logger.info(
                        "[%s] Epidermis: kept 1 of %d polygons (nearest centroid to image center)",
                        img_name,
                        len(epi_all),
                    )
                polys_by_class[YOLO_CLASS_EPIDERMIS] = epi_one

            label_path = os.path.join(self._labels_dir, f"{_stem(img_path)}.txt")
            write_yolo_seg_label_txt(label_path, polys_by_class)

            rgb = np.array(Image.open(img_path).convert("RGB"))
            pr_ring = app_params_for_stage(img_name, "ring", DEFAULT_STAGE)
            pr_casp = app_params_for_stage(img_name, "casp", DEFAULT_STAGE)
            roi_ring = build_ring_roi(h, w, polys_by_class)
            roi_casp = build_casp_inside_mask(h, w, polys_by_class)
            noise_ring = detect_noise_mask(rgb, roi_ring, pr_ring)
            noise_casp = detect_noise_mask(rgb, roi_casp, pr_casp)
            # Bitwise OR merges ring + casp noise for class-2 YOLO labels and area metrics.
            noise_union = np.clip(noise_ring.astype(np.uint8) | noise_casp.astype(np.uint8), 0, 1)

            merged_geom = os.path.join(self._geom_dir, f"{_stem(img_path)}_geometry_merged.txt")
            write_merged_label_with_noise(merged_geom, polys_by_class, noise_union, h, w)

            metrics = compute_stem_metric_row(
                h, w, polys_by_class, noise_union, vb_polys, main_poly, self.PIXEL_TO_MICRON
            )
            metrics["image_name"] = img_name
            metrics["notes"] = ""
            records.append(metrics)

            out_png = os.path.join(self.viz_output_dir, img_name)
            epi_polys = polys_by_class.get(YOLO_CLASS_EPIDERMIS, [])
            _draw_dual_viz(img_bgr, main_poly, vb_polys, noise_union, out_png, epidermis_polys=epi_polys)
            self.visualization_paths.append(out_png)
            _post_logger.info("[%s] vb_count=%s ring_um2=%s", img_name, metrics.get("vb_count"), metrics.get("ring_area_um2"))

        return pd.DataFrame(records)
