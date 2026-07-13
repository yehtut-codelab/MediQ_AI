"""Pathway prediction: presentation -> visit archetype -> empirical station sequence.

Archetypes map to sequences actually observed in the wait time export (mined by
scripts/mine_pathways.py into data/pathways.json). Per-station expected wait
and service durations come from the mined station statistics, so the projected
timeline is grounded in the same evidence base as the wait estimates.

Classification is keyword-based and ordered most-specific-first; a P1 triage
overrides everything to the acute walk-in pathway.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

PATHWAYS_PATH = Path(__file__).resolve().parents[2] / "data" / "pathways.json"

DILATION_MIN = 30.0  # standard mydriatic onset wait before dilated exam

# (compiled keyword pattern, archetype key) — first match wins, so order
# most-specific presentation first and generic reviews last.
_ARCHETYPE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"inject|anti.?vegf|ivt|lucentis|eylea|avastin", re.I), "intravitreal_treatment"),
    (re.compile(r"laser|yag|capsulotom|iridotom|slt\b", re.I), "laser_procedure"),
    (re.compile(r"refract|spectacle|prescription|presbyopia|myopia", re.I), "refraction_visit"),
    (re.compile(r"glaucoma|iop|intraocular pressure|visual field|hvf|cup.?disc", re.I), "glaucoma_review"),
    (re.compile(r"retinopathy|diabet|macula|amd\b|retina|oct\b|edema|drusen", re.I), "retina_imaging_review"),
    (re.compile(r"cataract.{0,30}(surger|list|biometry|pre.?op)|biometry", re.I), "cataract_listing"),
    (re.compile(r"scan|angiograph|imaging|photo", re.I), "imaging_workup"),
]

_DEFAULT_ARCHETYPE = "routine_review"
_ACUTE_ARCHETYPE = "acute_walkin"

# Archetype -> observed sequence + support (patient-day journeys in the export).
# Support counts from scripts/mine_pathways.py over the full dataset.
ARCHETYPES: dict[str, dict[str, Any]] = {
    "acute_walkin": {
        "label": "Acute walk-in assessment",
        "sequence": ["VA", "Consultation"],
        "support": 12245,
    },
    "glaucoma_review": {
        # HRT optic-nerve imaging dominates HVF in this dataset (338 vs 90 journeys)
        "label": "Glaucoma review (optic nerve imaging + consult)",
        "sequence": ["VA", "HRT", "Consultation"],
        "support": 338,
    },
    "retina_imaging_review": {
        "label": "Retina review with OCT (dilated)",
        "sequence": ["VA", "OCT", "Consultation"],
        "support": 2509,
        "dilation_before": "OCT",
    },
    "laser_procedure": {
        "label": "Laser procedure visit",
        "sequence": ["EYE-Laser", "VA", "Consultation"],
        "support": 1326,
    },
    "refraction_visit": {
        "label": "Refraction + consult",
        "sequence": ["REFRACTION", "Consultation"],
        "support": 1208,
    },
    "intravitreal_treatment": {
        "label": "Injection / treatment visit",
        "sequence": ["Pre Consultation Test", "Treatment"],
        "support": 2315,
    },
    "cataract_listing": {
        "label": "Cataract surgery workup (biometry + listing)",
        "sequence": ["Biometry", "PAT", "MO Clerking"],
        "support": 263,
    },
    "imaging_workup": {
        "label": "Diagnostic imaging + consult",
        "sequence": ["Pre Consultation Test", "Diagnostic", "Consultation"],
        "support": 7846,
    },
    "routine_review": {
        "label": "Routine review",
        "sequence": ["Pre Consultation Test", "Consultation"],
        "support": 16722,
    },
}


@lru_cache(maxsize=1)
def _mined() -> dict:
    return json.loads(PATHWAYS_PATH.read_text())


def classify(*texts: str | None, priority: int = 3) -> str:
    if priority == 1:
        return _ACUTE_ARCHETYPE
    blob = " ".join(t for t in texts if t)
    for pattern, archetype in _ARCHETYPE_RULES:
        if pattern.search(blob):
            return archetype
    return _DEFAULT_ARCHETYPE


def build_pathway(archetype: str) -> dict[str, Any]:
    """Projected station-by-station timeline with cumulative ETA offsets."""
    spec = ARCHETYPES[archetype]
    station_stats = _mined()["station_stats"]
    steps, offset = [], 0.0
    for station in spec["sequence"]:
        stats = station_stats.get(station, {})
        wait = float(stats.get("median_wait_min", 10.0))
        service = float(stats.get("median_contact_min", 5.0))
        note = None
        if spec.get("dilation_before") == station:
            offset += DILATION_MIN
            note = f"includes {DILATION_MIN:.0f} min dilation wait"
        steps.append({
            "station": station,
            "eta_offset_min": round(offset, 1),
            "expected_wait_min": wait,
            "expected_service_min": service,
            "note": note,
        })
        offset += wait + service
    return {
        "archetype": archetype,
        "label": spec["label"],
        "support": spec["support"],
        "steps": steps,
        "total_visit_min": round(offset, 1),
    }
