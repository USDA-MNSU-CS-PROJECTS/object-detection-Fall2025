"""PG/RR profiles and detection parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Stage = Literal["ring", "casp"]


@dataclass(frozen=True)
class DetectionParams:
    threshold: float = 80.0
    n_clusters: int = 8
    min_area: int = 0
    min_elongation: float = 0.0
    min_compactness: float = 0.0


def image_kind_from_filename(name: str) -> Literal["PG", "RR", "default"]:
    if "_PG" in name:
        return "PG"
    if "_RR" in name:
        return "RR"
    return "default"


def params_for_stage(filename: str, stage: Stage, cli: DetectionParams) -> DetectionParams:
    kind = image_kind_from_filename(filename)
    if kind == "PG":
        if stage == "ring":
            return DetectionParams(
                threshold=20.0,
                n_clusters=cli.n_clusters,
                min_area=200,
                min_elongation=cli.min_elongation,
                min_compactness=cli.min_compactness,
            )
        return DetectionParams(
            threshold=20.0,
            n_clusters=cli.n_clusters,
            min_area=25000,
            min_elongation=cli.min_elongation,
            min_compactness=cli.min_compactness,
        )
    if kind == "RR":
        if stage == "ring":
            return DetectionParams(
                threshold=50.0,
                n_clusters=cli.n_clusters,
                min_area=300,
                min_elongation=cli.min_elongation,
                min_compactness=cli.min_compactness,
            )
        return DetectionParams(
            threshold=50.0,
            n_clusters=cli.n_clusters,
            min_area=25000,
            min_elongation=cli.min_elongation,
            min_compactness=cli.min_compactness,
        )
    return cli
