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
from shapely.geometry import Point, Polygon
from shapely.validation import make_valid

from noise_deletion_clean.detection import detect_noise_mask
from noise_deletion_clean.masks import build_casp_inside_mask, build_ring_roi, polygons_to_pixel_mask

from config.inference_constants import (
    CLASS_CASPARIAN,
    CLASS_CASPARIAN_FALLBACK,
    CLASS_EPIDERMIS,
    CLASS_VASCULAR_BUNDLES,
    MODEL_A_EPIDERMIS_KEEP_NEAREST_TO_CENTER,
    MODEL_A_SWAP_EPI_CASP_LABELS,
    PIPELINE_EXPORT_METRIC_DEBUG_VIZ,
    PIPELINE_WRITE_OVERLAY_VISUALIZATIONS,
    PIXEL_TO_MICRON,
    YOLO_CLASS_CASPARIAN,
    YOLO_CLASS_EPIDERMIS,
)
from config.noise_profiles_app import DEFAULT_STAGE, app_params_for_stage
from filename_metadata import parse_image_filename_metadata
from stem_metrics import (
    build_na_row,
    compute_stem_metric_row,
    export_metric_debug_visualizations,
    order_result_dataframe,
)
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


def _shapely_polygon_from_normalized(
    pts: list[tuple[float, float]],
    h: int,
    w: int,
) -> Polygon | None:
    if len(pts) < 3:
        return None
    wx = max(w - 1, 1)
    hx = max(h - 1, 1)
    pts_px = [(float(x) * wx, float(y) * hx) for x, y in pts]
    poly = Polygon(pts_px)
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty:
        return None
    if poly.geom_type == "Polygon":
        return poly
    geoms = [g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon" and not g.is_empty]
    return max(geoms, key=lambda g: g.area) if geoms else None


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


def _filter_casparian_to_single_inside_epidermis(
    casp_norm_polys: list[list[tuple[float, float]]],
    epi_norm_poly: list[tuple[float, float]],
    h: int,
    w: int,
    img_name: str,
) -> list[list[tuple[float, float]]]:
    """Keep one Casparian: prefer centroids inside chosen epidermis; else nearest to image center."""
    epi = _shapely_polygon_from_normalized(epi_norm_poly, h, w)
    if epi is None:
        return []
    img_center = np.array([w / 2.0, h / 2.0])
    scored_inside: list[tuple[float, list[tuple[float, float]]]] = []
    scored_all: list[tuple[float, list[tuple[float, float]]]] = []
    for pts in casp_norm_polys:
        if len(pts) < 3:
            continue
        c_poly = _shapely_polygon_from_normalized(pts, h, w)
        if c_poly is None:
            continue
        cx, cy = float(c_poly.centroid.x), float(c_poly.centroid.y)
        dist = float(np.linalg.norm(img_center - np.array([cx, cy])))
        scored_all.append((dist, pts))
        if epi.covers(Point(cx, cy)):
            scored_inside.append((dist, pts))
    pool = scored_inside if scored_inside else scored_all
    if not pool:
        return []
    if not scored_inside and scored_all:
        _post_logger.info(
            "[%s] Casparian: no centroid inside epidermis; using nearest-to-center among all",
            img_name,
        )
    pool.sort(key=lambda t: t[0])
    return [pool[0][1]]


def _filter_vascular_bundles(epi_polygon: Polygon | None, preds):
    if epi_polygon is None or epi_polygon.is_empty:
        return []
    vb_polys: list = []
    for p in preds:
        if p["class_name"] != CLASS_VASCULAR_BUNDLES:
            continue
        cx, cy = _polygon_center(p["points"])
        if epi_polygon.covers(Point(cx, cy)):
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
        if PIPELINE_WRITE_OVERLAY_VISUALIZATIONS:
            os.makedirs(self.viz_output_dir, exist_ok=True)
        self.PIXEL_TO_MICRON = pixel_to_micron if pixel_to_micron is not None else PIXEL_TO_MICRON
        self.visualization_paths: list[str] = []
        log_dir = debug_log_dir if debug_log_dir else output_dir
        self._log_path = os.path.join(log_dir, "post_processor_debug.log")
        if debug_log_dir:
            os.makedirs(debug_log_dir, exist_ok=True)
        self._labels_dir = os.path.join(output_dir, "labels_generated")
        self._geom_dir = os.path.join(output_dir, "geometry_export")
        self._metric_debug_dir = os.path.join(output_dir, "metric_debug_viz")
        os.makedirs(self._labels_dir, exist_ok=True)
        os.makedirs(self._geom_dir, exist_ok=True)
        if PIPELINE_EXPORT_METRIC_DEBUG_VIZ:
            os.makedirs(self._metric_debug_dir, exist_ok=True)

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

            preds_b = _collect_polygons(rb)

            cmap = _class_map_model_a(ra)
            if not cmap:
                _post_logger.warning("Model A has no expected class names in %s", img_name)
                records.append(build_na_row(img_name, "Model A class names mismatch"))
                continue

            polys_by_class = result_to_polygons_by_class(ra, cmap, img_bgr.shape)
            if YOLO_CLASS_CASPARIAN not in polys_by_class or not polys_by_class.get(YOLO_CLASS_CASPARIAN):
                _post_logger.warning("No Casparian polygons after export in %s", img_name)
                records.append(build_na_row(img_name, "No Casparian mask"))
                continue
            if YOLO_CLASS_EPIDERMIS not in polys_by_class or not polys_by_class.get(YOLO_CLASS_EPIDERMIS):
                _post_logger.warning(
                    "No Epidermis polygons in %s (ring ROI needs class %s)",
                    img_name,
                    YOLO_CLASS_EPIDERMIS,
                )
                records.append(build_na_row(img_name, "No Epidermis mask"))
                continue

            h, w = img_bgr.shape[:2]
            epi_all = polys_by_class[YOLO_CLASS_EPIDERMIS]
            if len(epi_all) > 1:
                if not MODEL_A_EPIDERMIS_KEEP_NEAREST_TO_CENTER:
                    _post_logger.info(
                        "[%s] Epidermis: %d polygons; collapsing to one (nearest to center) for single-stem pipeline",
                        img_name,
                        len(epi_all),
                    )
                epi_one = _filter_epidermis_polygons_nearest_to_center(epi_all, h, w)
                if not epi_one:
                    _post_logger.warning("Epidermis nearest-to-center filter left no polygons in %s", img_name)
                    records.append(build_na_row(img_name, "Epidermis filter empty"))
                    continue
                if MODEL_A_EPIDERMIS_KEEP_NEAREST_TO_CENTER:
                    _post_logger.info(
                        "[%s] Epidermis: kept 1 of %d polygons (nearest centroid to image center)",
                        img_name,
                        len(epi_all),
                    )
                polys_by_class[YOLO_CLASS_EPIDERMIS] = epi_one

            epi_norm = polys_by_class[YOLO_CLASS_EPIDERMIS][0]
            casp_all = list(polys_by_class[YOLO_CLASS_CASPARIAN])
            casp_one = _filter_casparian_to_single_inside_epidermis(
                casp_all, epi_norm, h, w, img_name
            )
            if not casp_one:
                _post_logger.warning("No Casparian polygons after epidermis filter in %s", img_name)
                records.append(build_na_row(img_name, "Casparian filter empty"))
                continue
            polys_by_class[YOLO_CLASS_CASPARIAN] = casp_one

            epi_polygon = _shapely_polygon_from_normalized(epi_norm, h, w)
            main_poly = _shapely_polygon_from_normalized(casp_one[0], h, w)
            vb_polys = _filter_vascular_bundles(epi_polygon, preds_b)
            if main_poly is None:
                _post_logger.warning("Invalid Casparian polygon geometry in %s", img_name)
                records.append(build_na_row(img_name, "Casparian geometry invalid"))
                continue

            label_path = os.path.join(self._labels_dir, f"{_stem(img_path)}.txt")
            write_yolo_seg_label_txt(label_path, polys_by_class)

            rgb = np.array(Image.open(img_path).convert("RGB"))
            pr_ring = app_params_for_stage(img_name, "ring", DEFAULT_STAGE)
            pr_casp = app_params_for_stage(img_name, "casp", DEFAULT_STAGE)
            roi_ring = build_ring_roi(
                h,
                w,
                polys_by_class,
                epidermis_class_id=YOLO_CLASS_EPIDERMIS,
                casparian_class_id=YOLO_CLASS_CASPARIAN,
            )
            roi_casp = build_casp_inside_mask(
                h,
                w,
                polys_by_class,
                casparian_class_id=YOLO_CLASS_CASPARIAN,
            )
            noise_ring = detect_noise_mask(rgb, roi_ring, pr_ring)
            noise_casp = detect_noise_mask(rgb, roi_casp, pr_casp)
            # Bitwise OR merges ring + casp noise for class-2 YOLO labels and area metrics.
            noise_union = np.clip(noise_ring.astype(np.uint8) | noise_casp.astype(np.uint8), 0, 1)
            epi_mask_arr = polygons_to_pixel_mask(h, w, polys_by_class[YOLO_CLASS_EPIDERMIS])
            noise_union = (noise_union & epi_mask_arr).astype(np.uint8)

            merged_geom = os.path.join(self._geom_dir, f"{_stem(img_path)}_geometry_merged.txt")
            write_merged_label_with_noise(merged_geom, polys_by_class, noise_union, h, w)

            metrics = compute_stem_metric_row(
                h, w, polys_by_class, noise_union, vb_polys, self.PIXEL_TO_MICRON
            )
            if PIPELINE_EXPORT_METRIC_DEBUG_VIZ:
                try:
                    export_metric_debug_visualizations(
                        debug_root=self._metric_debug_dir,
                        image_stem=_stem(img_path),
                        height=h,
                        width=w,
                        polygons_by_class=polys_by_class,
                        noise_union=noise_union,
                        vb_polys=vb_polys,
                        pixel_to_micron=self.PIXEL_TO_MICRON,
                    )
                except Exception as exc:
                    _post_logger.warning("[%s] metric_debug_viz failed: %s", img_name, exc)
            meta = parse_image_filename_metadata(img_name)
            parse_ok = meta.pop("parse_ok")
            notes = "" if parse_ok else "filename_parse_failed"
            records.append({**meta, **metrics, "notes": notes})

            if PIPELINE_WRITE_OVERLAY_VISUALIZATIONS:
                out_png = os.path.join(self.viz_output_dir, img_name)
                epi_polys = polys_by_class.get(YOLO_CLASS_EPIDERMIS, [])
                _draw_dual_viz(img_bgr, main_poly, vb_polys, noise_union, out_png, epidermis_polys=epi_polys)
                self.visualization_paths.append(out_png)
            _post_logger.info(
                "[%s] vb_count=%s Epi_casp_zone_um2=%s",
                img_name,
                metrics.get("Vb_count"),
                metrics.get("Epi_casp_zone_area_um2"),
            )

        return order_result_dataframe(pd.DataFrame(records))
