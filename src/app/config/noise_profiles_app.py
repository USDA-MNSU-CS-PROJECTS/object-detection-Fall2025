"""Noise detection parameters for PG/RR profiles and ring/casp stages - edit here."""

from __future__ import annotations

from noise_deletion_clean.profiles import DetectionParams, image_kind_from_filename

DEFAULT_STAGE = DetectionParams(
    threshold=80.0,
    n_clusters=8,
    min_area=0,
    min_elongation=0.0,
    min_compactness=0.0,
)

_PG_RING = DetectionParams(
    threshold=20.0,
    n_clusters=8,
    min_area=200,
    min_elongation=0.0,
    min_compactness=0.0,
)
_PG_CASP = DetectionParams(
    threshold=20.0,
    n_clusters=8,
    min_area=25000,
    min_elongation=0.0,
    min_compactness=0.0,
)
_RR_RING = DetectionParams(
    threshold=50.0,
    n_clusters=8,
    min_area=300,
    min_elongation=0.0,
    min_compactness=0.0,
)
_RR_CASP = DetectionParams(
    threshold=50.0,
    n_clusters=8,
    min_area=25000,
    min_elongation=0.0,
    min_compactness=0.0,
)


def app_params_for_stage(filename: str, stage: str, cli: DetectionParams | None = None) -> DetectionParams:
    cli = cli or DEFAULT_STAGE
    kind = image_kind_from_filename(filename)
    if kind == "PG":
        return _PG_RING if stage == "ring" else _PG_CASP
    if kind == "RR":
        return _RR_RING if stage == "ring" else _RR_CASP
    return cli
