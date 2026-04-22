"""
Geometric stem metrics: epidermis-Casparian ring, medial-axis mean thickness, areas with noise subtracted.
Adjust formulas primarily in this module.
"""

from __future__ import annotations

import os
import sys

_third = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party"))
if _third not in sys.path:
    sys.path.insert(0, _third)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure as sk_measure
from skimage.morphology import medial_axis

from config.inference_constants import (
    RING_MASK_OPEN_BEFORE_MEDIAL_ITERATIONS,
    RING_MEDIAL_SPUR_PRUNE_MAX_ITER,
    RING_QUALITY_CLOSURE_AREA_WARN_RATIO,
    RING_QUALITY_CLOSURE_ITERATIONS,
    RING_QUALITY_MEDIAL_BRANCH_WARN,
    RING_QUALITY_MIN_MAIN_COMPONENT_AREA_FRACTION,
)

from noise_deletion_clean.masks import build_ring_roi, polygons_to_pixel_mask

from filename_metadata import parse_image_filename_metadata


def _px_to_um2(area_px: float, pixel_to_micron: float) -> float:
    return area_px * (pixel_to_micron**2)


def _px_to_um(dist_px: float, pixel_to_micron: float) -> float:
    return dist_px * pixel_to_micron


def _count_medial_branch_pixels(skel: np.ndarray) -> int:
    """Medial-axis pixels with degree >= 3 on the 8-connected skeleton (junctions)."""
    sk = skel.astype(np.uint8)
    if not sk.any():
        return 0
    acc = ndimage.convolve(sk, np.ones((3, 3), dtype=np.int32), mode="constant") * sk
    return int(np.sum(acc >= 4))


_KERNEL_SKEL_NB8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)


def _prune_medial_spurs(skel: np.ndarray, max_iterations: int) -> np.ndarray:
    """Remove 8-connected endpoints (degree 1) repeatedly until none remain or max_iterations."""
    sk = skel.astype(bool).copy()
    n_it = max(0, int(max_iterations))
    if n_it == 0:
        return sk
    for _ in range(n_it):
        if not sk.any():
            break
        nb = ndimage.convolve(sk.astype(np.uint8), _KERNEL_SKEL_NB8, mode="constant")
        leaves = sk & (nb == 1)
        if not leaves.any():
            break
        sk[leaves] = False
    return sk


def mean_ring_thickness_medial_axis_px(
    ring_mask: np.ndarray,
) -> tuple[
    float | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Ring mask: optional opening, then Euclidean DT on same working mask as medial axis;
    medial axis, spur-pruned; mean thickness = mean over pruned medial pixels of 2 * DT(x).

    Returns:
        mean_thickness_px, dt, medial_pruned, ring_used_for_ma, medial_raw, ring_input
    """
    ring_input = ring_mask.astype(bool)
    if not ring_input.any():
        zb = np.zeros_like(ring_mask, dtype=bool)
        zd = np.zeros_like(ring_mask, dtype=np.float64)
        return None, zd, zb, zb, zb, ring_input

    r = ring_input.copy()
    for _ in range(max(0, RING_MASK_OPEN_BEFORE_MEDIAL_ITERATIONS)):
        r = ndimage.binary_opening(r)

    if not r.any():
        zd = np.zeros_like(ring_mask, dtype=np.float64)
        zb = np.zeros_like(ring_mask, dtype=bool)
        return None, zd, zb, zb, zb, ring_input

    dt = ndimage.distance_transform_edt(r.astype(np.uint8))
    sk_raw = medial_axis(r, return_distance=False)
    if sk_raw is None or not np.asarray(sk_raw).any():
        return None, dt, np.zeros_like(r, dtype=bool), r, np.zeros_like(r, dtype=bool), ring_input

    sk_raw = np.asarray(sk_raw, dtype=bool)
    sk_pruned = _prune_medial_spurs(sk_raw, RING_MEDIAL_SPUR_PRUNE_MAX_ITER)

    if not sk_pruned.any():
        return None, dt, sk_pruned, r, sk_raw, ring_input

    vals = dt[sk_pruned]
    vals = vals[vals > 0]
    if vals.size == 0:
        vals = dt[sk_pruned]
    if vals.size == 0:
        return None, dt, sk_pruned, r, sk_raw, ring_input

    return float(2.0 * np.mean(vals)), dt, sk_pruned, r, sk_raw, ring_input


def assess_epi_casp_ring_quality(
    ring_bool: np.ndarray,
    medial: np.ndarray,
    *,
    ring_for_multi_component: np.ndarray | None = None,
) -> tuple[str, str]:
    """
    Labels: ok | warning | degraded. Detail is semicolon-separated diagnostic tokens.
    """
    parts: list[str] = []
    warn = False
    degraded = False

    r_mc = ring_for_multi_component if ring_for_multi_component is not None else ring_bool
    if not r_mc.any():
        return "degraded", "empty_ring"

    labeled, nlab = ndimage.label(r_mc.astype(np.uint8))
    areas = [int(np.sum(labeled == k)) for k in range(1, nlab + 1)]
    total_a = sum(areas)

    if nlab > 1:
        parts.append(f"multi_component:{nlab}")
        largest = max(areas) if areas else 0
        frac = largest / total_a if total_a > 0 else 0.0
        parts.append(f"largest_frac:{frac:.3f}")
        degraded = True
        if frac < RING_QUALITY_MIN_MAIN_COMPONENT_AREA_FRACTION:
            parts.append("main_component_small_frac")

    if total_a > 0:
        closed = r_mc.astype(bool)
        for _ in range(RING_QUALITY_CLOSURE_ITERATIONS):
            closed = ndimage.binary_closing(closed)
        csum = int(closed.sum())
        rsum = int(r_mc.sum())
        if rsum > 0 and csum >= rsum * RING_QUALITY_CLOSURE_AREA_WARN_RATIO:
            parts.append(f"closure_area_ratio:{csum / rsum:.3f}")
            warn = True

    br = _count_medial_branch_pixels(medial)
    parts.append(f"medial_branch_pts:{br}")
    if br > RING_QUALITY_MEDIAL_BRANCH_WARN:
        warn = True

    if not medial.any():
        parts.append("medial_axis_empty")
        degraded = True

    detail = ";".join(parts) if parts else "none"
    if degraded:
        return "degraded", detail
    if warn:
        return "warning", detail
    return "ok", detail


def _mask_area_px(m: np.ndarray) -> float:
    return float(m.astype(np.uint8).sum())


def _r4(x: float | None) -> float | None:
    if x is None:
        return None
    return round(float(x), 4)


STEM_METRIC_KEYS: list[str] = [
    "Epi_casp_zone_area_um2",
    "Epi_casp_zone_area_minus_noise_um2",
    "Mean_epi_casp_distance_um",
    "Epi_casp_ring_quality",
    "Epi_casp_ring_quality_detail",
    "Casparian_inner_area_um2",
    "Casparian_inner_area_minus_noise_um2",
    "Epidermis_area_um2",
    "Epidermis_area_minus_noise_um2",
    "Vb_count",
    "Vb_total_area_um2",
    "Vb_total_area_minus_noise_um2",
    "Vb_mean_area_um2",
    "Vb_mean_area_minus_noise_um2",
    "Vb_area_inside_casparian_um2",
    "Vb_area_inside_casparian_minus_noise_um2",
    "Vb_area_outside_casparian_within_epidermis_um2",
    "Vb_area_outside_casparian_within_epidermis_minus_noise_um2",
    "Vb_total_to_casparian_inner_area_ratio",
    "Vb_inside_to_outside_casparian_ratio",
]

METADATA_CSV_KEYS: list[str] = [
    "ImageID",
    "Year",
    "LabID",
    "Date",
    "DateSuffix",
    "Incubation_h",
    "Magnification",
    "Stitch",
    "Code",
    "ImageType",
    "Filename",
]

RESULT_CSV_COLUMN_ORDER: list[str] = METADATA_CSV_KEYS + STEM_METRIC_KEYS + ["notes"]


def build_na_row(basename_with_ext: str, note: str) -> dict:
    meta = parse_image_filename_metadata(basename_with_ext)
    meta.pop("parse_ok", None)
    na = {k: "N/A" for k in STEM_METRIC_KEYS}
    return {**meta, **na, "notes": note}


def order_result_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ordered = [c for c in RESULT_CSV_COLUMN_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]


def _compute_metric_components(
    height: int,
    width: int,
    polygons_by_class: dict[int, list[list[tuple[float, float]]]],
    noise_union: np.ndarray,
    vb_polys: list,
    pixel_to_micron: float,
) -> tuple[
    dict,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    ring = build_ring_roi(height, width, polygons_by_class)
    noise_u = np.clip(noise_union.astype(np.uint8), 0, 1)
    noise_b = noise_u.astype(bool)

    epi = polygons_to_pixel_mask(height, width, polygons_by_class.get(1, []))
    casp = polygons_to_pixel_mask(height, width, polygons_by_class.get(0, []))

    ring_clean = ring.astype(np.uint8)
    ring_minus_noise = (ring_clean.astype(bool) & ~noise_b).astype(np.uint8)

    t_clean, dt_ring, medial, ring_for_ma, medial_raw, ring_input = mean_ring_thickness_medial_axis_px(
        ring_clean
    )
    q_level, q_detail = assess_epi_casp_ring_quality(
        ring_input, medial, ring_for_multi_component=ring_for_ma
    )

    casp_clean = casp.astype(np.uint8)
    epi_clean = epi.astype(np.uint8)
    casp_mn = (casp_clean.astype(bool) & ~noise_b).astype(np.uint8)
    epi_mn = (epi_clean.astype(bool) & ~noise_b).astype(np.uint8)

    casp_bool = casp_clean.astype(bool)
    epi_bool = epi_clean.astype(bool)

    casp_area_px = _mask_area_px(casp_clean)
    vb_count = len(vb_polys)
    wx = max(width - 1, 1)
    hx = max(height - 1, 1)

    def _vb_poly_normalized(points: list) -> list[tuple[float, float]]:
        return [(float(px) / wx, float(py) / hx) for px, py in points]

    vb_total_px = 0.0
    vb_total_mn_px = 0.0
    vb_inside_px = 0.0
    vb_inside_mn_px = 0.0
    vb_out_epi_px = 0.0
    vb_out_epi_mn_px = 0.0
    vb_mask_total = np.zeros((height, width), dtype=np.uint8)

    for p in vb_polys:
        vm = polygons_to_pixel_mask(height, width, [_vb_poly_normalized(p)]).astype(bool)
        vb_mask_total |= vm.astype(np.uint8)
        px = float(vm.sum())
        vb_total_px += px
        mn = float((vm & ~noise_b).sum())
        vb_total_mn_px += mn
        vb_inside_px += float((vm & casp_bool).sum())
        vb_inside_mn_px += float((vm & casp_bool & ~noise_b).sum())
        outside_in_epi = vm & epi_bool & ~casp_bool
        vb_out_epi_px += float(outside_in_epi.sum())
        vb_out_epi_mn_px += float((outside_in_epi & ~noise_b).sum())

    p2 = pixel_to_micron
    vb_mean_px = vb_total_px / vb_count if vb_count > 0 else 0.0
    vb_mean_mn_px = vb_total_mn_px / vb_count if vb_count > 0 else 0.0

    ratio_total_casp = None
    if casp_area_px > 0:
        ratio_total_casp = float(vb_total_px) / float(casp_area_px)

    ratio_in_out = None
    if vb_out_epi_px > 0:
        ratio_in_out = float(vb_inside_px) / float(vb_out_epi_px)

    row = {
        "Epi_casp_zone_area_um2": _r4(_px_to_um2(_mask_area_px(ring_clean), p2)),
        "Epi_casp_zone_area_minus_noise_um2": _r4(_px_to_um2(_mask_area_px(ring_minus_noise), p2)),
        "Mean_epi_casp_distance_um": _r4(_px_to_um(t_clean, p2)) if t_clean is not None else None,
        "Epi_casp_ring_quality": q_level,
        "Epi_casp_ring_quality_detail": q_detail,
        "Casparian_inner_area_um2": _r4(_px_to_um2(casp_area_px, p2)),
        "Casparian_inner_area_minus_noise_um2": _r4(_px_to_um2(_mask_area_px(casp_mn), p2)),
        "Epidermis_area_um2": _r4(_px_to_um2(_mask_area_px(epi_clean), p2)),
        "Epidermis_area_minus_noise_um2": _r4(_px_to_um2(_mask_area_px(epi_mn), p2)),
        "Vb_count": vb_count,
        "Vb_total_area_um2": _r4(_px_to_um2(vb_total_px, p2)),
        "Vb_total_area_minus_noise_um2": _r4(_px_to_um2(vb_total_mn_px, p2)),
        "Vb_mean_area_um2": _r4(_px_to_um2(vb_mean_px, p2)),
        "Vb_mean_area_minus_noise_um2": _r4(_px_to_um2(vb_mean_mn_px, p2)) if vb_count > 0 else None,
        "Vb_area_inside_casparian_um2": _r4(_px_to_um2(vb_inside_px, p2)),
        "Vb_area_inside_casparian_minus_noise_um2": _r4(_px_to_um2(vb_inside_mn_px, p2)),
        "Vb_area_outside_casparian_within_epidermis_um2": _r4(_px_to_um2(vb_out_epi_px, p2)),
        "Vb_area_outside_casparian_within_epidermis_minus_noise_um2": _r4(_px_to_um2(vb_out_epi_mn_px, p2)),
        "Vb_total_to_casparian_inner_area_ratio": _r4(ratio_total_casp) if ratio_total_casp is not None else None,
        "Vb_inside_to_outside_casparian_ratio": _r4(ratio_in_out) if ratio_in_out is not None else None,
    }

    masks = {
        "noise_union": noise_u,
        "ring_clean": ring_clean,
        "ring_minus_noise": ring_minus_noise.astype(np.uint8),
        "casp_clean": casp_clean,
        "casp_minus_noise": casp_mn,
        "epi_clean": epi_clean,
        "epi_minus_noise": epi_mn,
        "vb_total": vb_mask_total,
        "vb_total_minus_noise": (vb_mask_total.astype(bool) & ~noise_b).astype(np.uint8),
        "vb_inside_casp": (vb_mask_total.astype(bool) & casp_bool).astype(np.uint8),
        "vb_inside_casp_minus_noise": (vb_mask_total.astype(bool) & casp_bool & ~noise_b).astype(np.uint8),
        "vb_outside_casp_within_epi": (vb_mask_total.astype(bool) & epi_bool & ~casp_bool).astype(np.uint8),
        "vb_outside_casp_within_epi_minus_noise": (
            vb_mask_total.astype(bool) & epi_bool & ~casp_bool & ~noise_b
        ).astype(np.uint8),
    }
    return row, masks, ring_clean, dt_ring, medial, medial_raw, ring_for_ma, ring_input, epi_bool, casp_bool


def export_metric_debug_visualizations(
    debug_root: str,
    image_stem: str,
    height: int,
    width: int,
    polygons_by_class: dict[int, list[list[tuple[float, float]]]],
    noise_union: np.ndarray,
    vb_polys: list,
    pixel_to_micron: float,
) -> None:
    os.makedirs(debug_root, exist_ok=True)
    out_dir = os.path.join(debug_root, image_stem)
    os.makedirs(out_dir, exist_ok=True)

    row, masks, ring_clean, dt_ring, medial, medial_raw, ring_for_ma, ring_input, epi_bool, casp_bool = (
        _compute_metric_components(
            height, width, polygons_by_class, noise_union, vb_polys, pixel_to_micron
        )
    )

    metric_mask_map = {
        "Epi_casp_zone_area_um2": masks["ring_clean"],
        "Epi_casp_zone_area_minus_noise_um2": masks["ring_minus_noise"],
        "Casparian_inner_area_um2": masks["casp_clean"],
        "Casparian_inner_area_minus_noise_um2": masks["casp_minus_noise"],
        "Epidermis_area_um2": masks["epi_clean"],
        "Epidermis_area_minus_noise_um2": masks["epi_minus_noise"],
        "Vb_total_area_um2": masks["vb_total"],
        "Vb_total_area_minus_noise_um2": masks["vb_total_minus_noise"],
        "Vb_area_inside_casparian_um2": masks["vb_inside_casp"],
        "Vb_area_inside_casparian_minus_noise_um2": masks["vb_inside_casp_minus_noise"],
        "Vb_area_outside_casparian_within_epidermis_um2": masks["vb_outside_casp_within_epi"],
        "Vb_area_outside_casparian_within_epidermis_minus_noise_um2": masks["vb_outside_casp_within_epi_minus_noise"],
        "noise_union": masks["noise_union"],
    }

    for metric_name, metric_mask in metric_mask_map.items():
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        ax.imshow(metric_mask, cmap="viridis")
        ax.set_title(f"{metric_name}: {row.get(metric_name, 'mask_only')}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{metric_name}.png"), bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    ring_viz = ring_input.astype(bool)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(dt_ring, cmap="magma")
    yr, xr = np.where(medial_raw)
    if len(xr) > 0:
        ax.scatter(xr, yr, s=1, c="white", alpha=0.35)
    y, x = np.where(medial)
    if len(x) > 0:
        ax.scatter(x, y, s=2, c="cyan", alpha=0.8)
    ax.set_title(
        f"Mean_epi_casp_distance_um: {row.get('Mean_epi_casp_distance_um')} | "
        f"{row.get('Epi_casp_ring_quality')} | {row.get('Epi_casp_ring_quality_detail')}"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Mean_epi_casp_distance_um.png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # Overlay: epidermis boundary, Casparian boundary, ring fill, medial axis on DT
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[..., 0] = epi_bool.astype(np.float32) * 0.25
    base[..., 1] = casp_bool.astype(np.float32) * 0.35
    base[..., 2] = ring_viz.astype(np.float32) * 0.2
    dt_norm = dt_ring.astype(np.float64)
    rb = ring_for_ma.astype(bool)
    if rb.any() and np.isfinite(dt_norm).any() and float(dt_ring[rb].max()) > 0:
        p99 = float(np.percentile(dt_ring[rb], 99))
        dt_norm = dt_norm / (p99 + 1e-9)
    dt_norm = np.clip(dt_norm, 0, 1)
    overlay = np.stack([dt_norm * 0.9, dt_norm * 0.6, dt_norm * 0.95], axis=-1)
    blend = 0.45 * base + 0.55 * overlay
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(blend)
    for contour in sk_measure.find_contours(epi_bool.astype(np.float64), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color="red", linewidth=1.2, alpha=0.9)
    for contour in sk_measure.find_contours(casp_bool.astype(np.float64), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color="blue", linewidth=1.2, alpha=0.9)
    my0, mx0 = np.where(medial_raw)
    if len(mx0) > 0:
        ax.scatter(mx0, my0, s=1, c="white", alpha=0.25)
    my, mx = np.where(medial)
    if len(mx) > 0:
        ax.scatter(mx, my, s=2, c="lime", alpha=0.9)
    ax.set_title(
        f"Ring medial overlay | quality={row.get('Epi_casp_ring_quality')}\n"
        f"{row.get('Epi_casp_ring_quality_detail')}"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Epi_casp_ring_medial_overlay.png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    scalar_metrics = [
        "Vb_count",
        "Vb_mean_area_um2",
        "Vb_mean_area_minus_noise_um2",
        "Vb_total_to_casparian_inner_area_ratio",
        "Vb_inside_to_outside_casparian_ratio",
        "Epi_casp_ring_quality",
        "Epi_casp_ring_quality_detail",
    ]
    for metric_name in scalar_metrics:
        fig, ax = plt.subplots(figsize=(6, 2.5), dpi=120)
        ax.text(0.02, 0.5, f"{metric_name} = {row.get(metric_name)}", fontsize=12, va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{metric_name}.png"), bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)


def compute_stem_metric_row(
    height: int,
    width: int,
    polygons_by_class: dict[int, list[list[tuple[float, float]]]],
    noise_union: np.ndarray,
    vb_polys: list,
    pixel_to_micron: float,
) -> dict:
    row, _masks, _r, _d, _m, _mr, _rf, _ri, _e, _c = _compute_metric_components(
        height, width, polygons_by_class, noise_union, vb_polys, pixel_to_micron
    )
    return row
