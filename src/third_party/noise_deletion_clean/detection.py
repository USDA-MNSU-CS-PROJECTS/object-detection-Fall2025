"""Dominant color and noise mask."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

from .profiles import DetectionParams


def get_dominant_color(rgb: np.ndarray, n_clusters: int = 8) -> np.ndarray:
    pixels = rgb.reshape(-1, 3).astype(np.float64)
    n_sample = min(50_000, len(pixels))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(pixels), size=n_sample, replace=False)
    sample = pixels[indices]
    n_clusters_actual = min(n_clusters, len(np.unique(sample, axis=0)))
    kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
    kmeans.fit(sample)
    labels = kmeans.predict(pixels)
    counts = np.bincount(labels, minlength=n_clusters_actual)
    dominant_label = int(np.argmax(counts))
    return kmeans.cluster_centers_[dominant_label].astype(np.float64)


def color_distance_mask(rgb: np.ndarray, dominant: np.ndarray, threshold: float) -> np.ndarray:
    diff = rgb.astype(np.float64) - dominant
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return (dist <= threshold).astype(np.uint8)


def filter_mask_by_min_area(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    sizes = np.bincount(labeled.ravel(), minlength=num_features + 1)[1:]
    keep = sizes >= min_area
    keep_labels = np.where(keep)[0] + 1
    out = np.zeros_like(mask)
    for lab in keep_labels:
        out[labeled == lab] = 1
    return out


def filter_mask_by_shape(
    mask: np.ndarray,
    min_elongation: float = 0.0,
    min_compactness: float = 0.0,
) -> np.ndarray:
    if min_elongation <= 0 and min_compactness <= 0:
        return mask
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    slices = ndimage.find_objects(labeled)
    keep_labels: list[int] = []
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        lab = i + 1
        comp_crop = labeled[sl] == lab
        area = int(comp_crop.sum())
        if area == 0:
            continue
        dy, dx = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        elongation = max(dy, dx) / max(min(dy, dx), 1)
        eroded = ndimage.binary_erosion(comp_crop, structure=cross)
        perimeter = int((comp_crop & ~eroded).sum())
        if perimeter == 0:
            compactness = 0.0
        else:
            compactness = (4 * math.pi * area) / (perimeter * perimeter)
        if (min_elongation <= 0 or elongation >= min_elongation) and (
            min_compactness <= 0 or compactness >= min_compactness
        ):
            keep_labels.append(lab)
    out = np.zeros_like(mask)
    for lab in keep_labels:
        out[labeled == lab] = 1
    return out


def detect_noise_mask(rgb: np.ndarray, roi_mask: np.ndarray | None, params: DetectionParams) -> np.ndarray:
    """
    Full pipeline: dominant color -> threshold -> min_area -> shape filters -> intersect with roi_mask if given.
    roi_mask: uint8 0/1, same HxW as rgb.
    """
    dom = get_dominant_color(rgb, n_clusters=params.n_clusters)
    m = color_distance_mask(rgb, dom, params.threshold)
    m = filter_mask_by_min_area(m, params.min_area)
    m = filter_mask_by_shape(m, params.min_elongation, params.min_compactness)
    if roi_mask is not None:
        m = m & roi_mask
    return m.astype(np.uint8)
