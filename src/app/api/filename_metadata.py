"""
Parse experiment metadata from image filenames (stem convention).
"""

from __future__ import annotations

import os
import re
from typing import Any

_STEM_RE = re.compile(
    r"^([0-9a-f]{8})-(\d{8})([a-z])_T(\d+)_(\d+x)([A-Za-z]+)_([A-Z]+)(?:_(\d+))?$"
)


def parse_image_filename_metadata(basename_with_ext: str) -> dict[str, Any]:
    """
    Expects basename with extension, e.g. '0afc45e2-20240705d_T24_10xstitch_RR.png'.
    On mismatch, returns empty strings for structured fields and parse_ok=False.
    """
    stem = os.path.splitext(basename_with_ext)[0]
    m = _STEM_RE.match(stem)
    out: dict[str, Any] = {
        "ImageID": stem,
        "Filename": basename_with_ext,
        "Year": "",
        "LabID": "",
        "Date": "",
        "DateSuffix": "",
        "Incubation_h": "",
        "Magnification": "",
        "Stitch": "",
        "Code": "",
        "ImageType": "",
        "parse_ok": False,
    }
    if not m:
        return out

    _prefix, ymd, letter, tnum, mag, stitch, code, _opt = m.groups()
    out["ImageID"] = stem
    out["Year"] = ymd[:4]
    out["Date"] = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
    out["DateSuffix"] = letter
    out["Incubation_h"] = int(tnum)
    out["Magnification"] = mag
    out["Stitch"] = stitch
    out["Code"] = code
    out["ImageType"] = f"{mag}{stitch}"
    out["parse_ok"] = True
    return out
