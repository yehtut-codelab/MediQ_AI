"""Load the TTSH wait time Excel export and clean it per spec DQ-1..DQ-5."""

import re
from pathlib import Path

import holidays
import pandas as pd

SGT = "Asia/Singapore"

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_sg_holidays = holidays.Singapore(years=range(2025, 2028))

# DQ-4: strip clinic prefixes so `EYE-VA Room 51` and `VA Room 51` collapse together
_PREFIX_RE = re.compile(r"^(EYE\s*-\s*|EYE-|C1A-|C1A\s*-\s*)", flags=re.IGNORECASE)


def normalize_service_point(sp: str) -> str:
    sp = _PREFIX_RE.sub("", str(sp).strip())
    return re.sub(r"\s+", " ", sp)


def daypart(hour: int) -> str:
    if hour < 9:
        return "early morning"
    if hour < 12:
        return "mid-morning"
    if hour < 14:
        return "lunchtime"
    if hour < 17:
        return "afternoon"
    return "evening"


def load_clean_events(path: Path, limit: int | None = None,
                      max_wait_sec: float = 14_400.0) -> pd.DataFrame:
    df = pd.read_excel(path)

    # DQ-1 / DQ-3
    df = df.dropna(subset=["Wait Start Date Time", "Waiting Time in sec", "Service Type"])
    # DQ-2
    df = df[(df["Waiting Time in sec"] >= 0) & (df["Waiting Time in sec"] <= max_wait_sec)]

    # DQ-5: timestamps are naive local times in the export — localize to SGT
    wait_start = pd.to_datetime(df["Wait Start Date Time"]).dt.tz_localize(
        SGT, ambiguous="NaT", nonexistent="NaT"
    )
    df = df.assign(wait_start=wait_start).dropna(subset=["wait_start"])

    df["service_point_norm"] = df["Service Point"].fillna("Unknown").map(normalize_service_point)
    df["hour"] = df["wait_start"].dt.hour
    df["day_of_week"] = df["wait_start"].dt.dayofweek  # 0=Monday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["wait_start"].dt.month
    df["is_public_holiday"] = (
        df["wait_start"].dt.date.map(lambda d: d in _sg_holidays).astype(int)
    )
    df["wait_min"] = (df["Waiting Time in sec"] / 60).round(2)
    df["contact_min"] = (df["Total Contact Time in sec"] / 60).round(2)

    df = df.sort_values("wait_start").reset_index(drop=True)
    if limit:
        df = df.head(limit)
    return df


def event_sentence(row: pd.Series) -> str:
    """Canonical text embedded for each station-visit event (spec §3.1)."""
    dow = DAY_NAMES[int(row["day_of_week"])]
    holiday = " on a public holiday" if row["is_public_holiday"] else ""
    return (
        f"{row['Service Type']} at {row['service_point_norm']}, {row['Clinic Name']}, "
        f"on {dow} at {int(row['hour']):02d}:00{holiday}, month {int(row['month'])}, "
        f"{daypart(int(row['hour']))} {'weekend' if row['is_weekend'] else 'weekday'}; "
        f"waited {row['wait_min']:.0f} minutes, service took {row['contact_min']:.0f} minutes."
    )
