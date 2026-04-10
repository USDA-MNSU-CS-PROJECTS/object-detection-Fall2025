"""
Geometric stem metrics: epidermis-Casparian ring, skeleton-based thickness, areas with noise subtracted.
Adjust formulas primarily in this module.
"""

from __future__ import annotations

import os
import sys

_third = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party"))
if _third not in sys.path:
    sys.path.insert(0, _third)

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon
from shapely.validation import make_valid
from shapely.ops import unary_union
from skimage.morphology import skeletonize

from noise_deletion_clean.masks import build_ring_roi, polygons_to_pixel_mask


def _px_to_um2(area_px: float, pixel_to_micron: float) -> float:
    return area_px * (pixel_to_micron**2)


def _px_to_um(dist_px: float, pixel_to_micron: float) -> float:
    return dist_px * pixel_to_micron


def mean_ring_thickness_from_skeleton_px(ring_mask: np.ndarray) -> float | None:
    """
    Ring mask R: skeletonize R; at each skeleton pixel take distance to nearest ring boundary
    (via distance transform); local thickness = 2 * distance; return mean over skeleton (pixels).
    """
    r = ring_mask.astype(bool)
    if not r.any():
        return None
    eroded = ndimage.binary_erosion(r)
    boundary = r & ~eroded
    if not boundary.any():
        return None
    input_dt = np.where(boundary, 0, 1).astype(np.uint8)
    dt = ndimage.distance_transform_edt(input_dt)
    sk = skeletonize(r)
    if not sk.any():
        return None
    vals = dt[sk]
    vals = vals[vals > 0]
    if vals.size == 0:
        vals = dt[sk]
    if vals.size == 0:
        return None
    return float(2.0 * np.mean(vals))


def _mask_area_px(m: np.ndarray) -> float:
    return float(m.astype(np.uint8).sum())


def compute_stem_metric_row(
    height: int,
    width: int,
    polygons_by_class: dict[int, list[list[tuple[float, float]]]],
    noise_union: np.ndarray,
    vb_polys: list,
    main_cs_poly: Polygon,
    pixel_to_micron: float,
) -> dict:
    ring = build_ring_roi(height, width, polygons_by_class)
    noise_u = np.clip(noise_union.astype(np.uint8), 0, 1)

    epi = polygons_to_pixel_mask(height, width, polygons_by_class.get(1, []))
    casp = polygons_to_pixel_mask(height, width, polygons_by_class.get(0, []))

    ring_clean = ring.astype(np.uint8)
    ring_minus_noise = (ring_clean.astype(bool) & (noise_u == 0)).astype(np.uint8)

    t_clean = mean_ring_thickness_from_skeleton_px(ring_clean)
    t_mn = mean_ring_thickness_from_skeleton_px(ring_minus_noise)

    casp_clean = casp.astype(np.uint8)
    epi_clean = epi.astype(np.uint8)
    casp_mn = (casp_clean.astype(bool) & (noise_u == 0)).astype(np.uint8)
    epi_mn = (epi_clean.astype(bool) & (noise_u == 0)).astype(np.uint8)

    vb_pixel_area = sum(float(Polygon(p).area) for p in vb_polys) if vb_polys else 0.0
    try:
        complete_cs = main_cs_poly
        if vb_polys:
            polys = [main_cs_poly]
            for p in vb_polys:
                poly = Polygon(p)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_valid and not poly.is_empty:
                    polys.append(poly)
            combined = unary_union(polys) if len(polys) > 1 else polys[0]
            if not combined.is_valid:
                combined = make_valid(combined)
            complete_cs = combined.convex_hull
        cs_pixel_area = float(complete_cs.area)
    except Exception:
        cs_pixel_area = float(main_cs_poly.area)

    ratio = vb_pixel_area / cs_pixel_area if cs_pixel_area > 0 else 0.0
    vb_count = len(vb_polys)
    avg_vb = vb_pixel_area / vb_count if vb_count > 0 else 0.0

    row = {
        "ring_area_um2": round(_px_to_um2(_mask_area_px(ring_clean), pixel_to_micron), 4),
        "ring_area_minus_noise_um2": round(_px_to_um2(_mask_area_px(ring_minus_noise), pixel_to_micron), 4),
        "mean_ring_thickness_um": round(_px_to_um(t_clean, pixel_to_micron), 4) if t_clean is not None else None,
        "mean_ring_thickness_minus_noise_um": round(_px_to_um(t_mn, pixel_to_micron), 4) if t_mn is not None else None,
        "casp_area_um2": round(_px_to_um2(_mask_area_px(casp_clean), pixel_to_micron), 4),
        "casp_area_minus_noise_um2": round(_px_to_um2(_mask_area_px(casp_mn), pixel_to_micron), 4),
        "epidermis_area_um2": round(_px_to_um2(_mask_area_px(epi_clean), pixel_to_micron), 4),
        "epidermis_area_minus_noise_um2": round(_px_to_um2(_mask_area_px(epi_mn), pixel_to_micron), 4),
        "vb_count": vb_count,
        "vb_area_microns": round(_px_to_um2(vb_pixel_area, pixel_to_micron), 4),
        "avg_vb_area_microns": round(_px_to_um2(avg_vb, pixel_to_micron), 4),
        "cs_area_microns": round(_px_to_um2(cs_pixel_area, pixel_to_micron), 4),
        "vb_to_cs_ratio": round(ratio, 4),
    }
    return row
