"""CLI check: nearest historical events for a hypothetical arriving patient.

Usage:
    python scripts/query_similar.py --clinic "TTSH Eye Centre" \
        --service-type Consultation --hour 9 --dow 3
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.embedder import embed_one
from app.services.preprocess import DAY_NAMES, daypart
from app.services.qdrant_service import build_filter, get_client, search_similar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic", default=None)
    parser.add_argument("--service-type", required=True)
    parser.add_argument("--hour", type=int, required=True)
    parser.add_argument("--dow", type=int, required=True, help="0=Monday .. 6=Sunday")
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    query = (
        f"{args.service_type} at {args.clinic or 'eye clinic'}, "
        f"on {DAY_NAMES[args.dow]} at {args.hour:02d}:00, "
        f"{daypart(args.hour)} {'weekend' if args.dow >= 5 else 'weekday'}."
    )
    print(f"Query: {query}\n")

    hits = search_similar(
        get_client(),
        embed_one(query),
        build_filter(clinic=args.clinic, service_type=args.service_type,
                     hour=args.hour, day_of_week=args.dow),
        limit=args.k,
    )
    if not hits:
        sys.exit("No similar events found — has the ingestion run? Try relaxing filters.")

    waits = [h["wait_min"] for h in hits]
    print(f"{len(hits)} nearest events | wait min: median {statistics.median(waits):.1f}, "
          f"mean {statistics.mean(waits):.1f}, max {max(waits):.1f}\n")
    for h in hits[:10]:
        print(f"  {h['score']:.3f}  {h['wait_start_iso'][:16]}  {h['clinic'][:16]:16} "
              f"{h['service_point'][:28]:28} wait {h['wait_min']:6.1f}m")


if __name__ == "__main__":
    main()
