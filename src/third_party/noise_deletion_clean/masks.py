"""ROI construction and polygon rasterization."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import find_contours


def polygons_to_pixel_mask(
    height: int,
    width: int,
    polygons: list[list[tuple[float, float]]],
) -> np.ndarray:
    if not polygons:
        return np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    pil_img = Image.fromarray(mask)
    draw = ImageDraw.Draw(pil_img)

    def to_pixel(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(x * (width - 1), y * (height - 1)) for x, y in poly]

    for poly in polygons:
        if len(poly) >= 3:
            draw.polygon(to_pixel(poly), fill=1)
    return np.array(pil_img)


def polygons_to_pixel_area(height: int, width: int, polygons: list[list[tuple[float, float]]]) -> int:
    return int(polygons_to_pixel_mask(height, width, polygons).sum())


def build_ring_roi(
    height: int,
    width: int,
    polygons_by_class: dict[int, list[list[tuple[float, float]]]],
) -> np.ndarray:
    """Ring: inside epidermis (class 1), outside Casparian (class 0)."""
    epi_polys = polygons_by_class.get(1, [])
    casp_polys = polygons_by_class.get(0, [])
    epi = np.zeros((height, width), dtype=np.uint8)
    casp = np.zeros((height, width), dtype=np.uint8)
    pil_epi = Image.fromarray(epi)
    pil_casp = Image.fromarray(casp)
    de = ImageDraw.Draw(pil_epi)
    dc = ImageDraw.Draw(pil_casp)

    def tp(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(x * (width - 1), y * (height - 1)) for x, y in poly]

    for poly in epi_polys:
        de.polygon(tp(poly), fill=1)
    for poly in casp_polys:
        dc.polygon(tp(poly), fill=1)
    epi = np.array(pil_epi)
    casp = np.array(pil_casp)
    return ((epi == 1) & (casp == 0)).astype(np.uint8)


def build_casp_inside_mask(
    height: int,
    width: int,
    polygons_by_class: dict[int, list[list[tuple[float, float]]]],
) -> np.ndarray:
    """ROI = filled Casparian polygons (class 0)."""
    casp_polys = polygons_by_class.get(0, [])
    casp = np.zeros((height, width), dtype=np.uint8)
    pil_casp = Image.fromarray(casp)
    dc = ImageDraw.Draw(pil_casp)

    def tp(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(x * (width - 1), y * (height - 1)) for x, y in poly]

    for poly in casp_polys:
        dc.polygon(tp(poly), fill=1)
    return np.array(pil_casp).astype(np.uint8)


def mask_to_normalized_polygons(mask: np.ndarray, height: int, width: int) -> list[list[tuple[float, float]]]:
    if mask.sum() == 0:
        return []
    contours = find_contours(mask.astype(np.float64), 0.5)
    out: list[list[tuple[float, float]]] = []
    for cont in contours:
        poly = [
            (
                float(col) / (width - 1) if width > 1 else 0.0,
                float(row) / (height - 1) if height > 1 else 0.0,
            )
            for row, col in cont
        ]
        if len(poly) >= 3:
            out.append(poly)
    return out
