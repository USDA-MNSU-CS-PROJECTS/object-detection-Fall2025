"""
Parse experiment metadata from image filenames (stem convention).
"""

from __future__ import annotations

import datetime as dt
import os
import re
from typing import Any

_STEM_RE = re.compile(
    r"^(\d{8})([a-z])_T(\d+)_(\d+x)([A-Za-z]+)_([A-Z]+)(?:_(\d+))?$"
)


def parse_image_filename_metadata(basename_with_ext: str) -> dict[str, Any]:
    """
    Expects basename with extension, e.g. '20230205a_T4_10xstitch_RR.nd2'.
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
        "parse_error": "filename_pattern_mismatch",
    }
    if not m:
        return out

    ymd, letter, tnum, mag, stitch, code, _opt = m.groups()
    out["ImageID"] = stem
    out["Year"] = ymd[:4]
    out["Date"] = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
    out["DateSuffix"] = letter
    out["Incubation_h"] = int(tnum)
    out["Magnification"] = mag
    out["Stitch"] = stitch
    out["Code"] = code
    out["ImageType"] = f"{mag}{stitch}"
    try:
        dt.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))
    except ValueError:
        out["parse_error"] = "invalid_calendar_date"
        return out

    out["parse_ok"] = True
    out["parse_error"] = ""
    return out
