"""Mine empirical patient pathways from the TTSH wait time export.

Groups station-visit events by (patient, day), orders them by wait start,
and aggregates the observed service-type sequences plus per-station wait and
contact statistics. Output feeds the pathway prediction service.

Usage:
    python scripts/mine_pathways.py                 # full dataset
    python scripts/mine_pathways.py --limit 50000   # smoke test
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.services.preprocess import load_clean_events

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "pathways.json"
TOP_SEQUENCES = 40


def collapse(seq: list[str]) -> tuple[str, ...]:
    """Drop consecutive repeats: VA, VA, Consultation -> VA, Consultation."""
    out: list[str] = []
    for s in seq:
        if not out or out[-1] != s:
            out.append(s)
    return tuple(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-file", type=Path, default=settings.data_file)
    args = parser.parse_args()

    t0 = time.time()
    print(f"Loading + cleaning {args.data_file.name} ...")
    df = load_clean_events(args.data_file, limit=args.limit,
                           max_wait_sec=settings.max_wait_sec)
    df["visit_date"] = df["wait_start"].dt.date

    print(f"  {len(df):,} events -> grouping into patient-day journeys ...")
    journeys = (
        df.sort_values("wait_start")
        .groupby(["Anonymized_ID", "visit_date"])["Service Type"]
        .agg(list)
    )
    sequences = Counter(collapse(seq) for seq in journeys)
    n_journeys = sum(sequences.values())

    station_stats = {
        st: {
            "median_wait_min": round(g["wait_min"].median(), 1),
            "p75_wait_min": round(g["wait_min"].quantile(0.75), 1),
            "median_contact_min": round(g["contact_min"].median(), 1),
            "n": int(len(g)),
        }
        for st, g in df.groupby("Service Type")
    }

    top = [
        {"sequence": list(seq), "count": c, "share": round(c / n_journeys, 4)}
        for seq, c in sequences.most_common(TOP_SEQUENCES)
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "mined_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_events": int(len(df)),
        "n_journeys": int(n_journeys),
        "station_stats": station_stats,
        "sequences": top,
    }, indent=2))

    print(f"\n{n_journeys:,} patient-day journeys, "
          f"{len(sequences):,} distinct sequences. Top 15:")
    for item in top[:15]:
        print(f"  {item['share']*100:5.1f}%  {item['count']:>6,}  "
              f"{' -> '.join(item['sequence'])}")
    print(f"\nWrote {OUT_PATH}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
