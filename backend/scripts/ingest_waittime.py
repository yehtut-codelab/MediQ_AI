"""Ingest the wait time Excel export into Qdrant.

Usage:
    python scripts/ingest_waittime.py                # full dataset
    python scripts/ingest_waittime.py --limit 5000   # smoke test
    python scripts/ingest_waittime.py --recreate     # drop & rebuild collection
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from app.config import settings
from app.services.embedder import get_model
from app.services.preprocess import event_sentence, load_clean_events
from app.services.qdrant_service import (
    collection_status,
    ensure_collection,
    get_client,
    upsert_events,
)

EMBED_CHUNK = 2048  # rows embedded+upserted per outer chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Max rows (smoke test)")
    parser.add_argument("--recreate", action="store_true", help="Drop and rebuild collection")
    parser.add_argument("--data-file", type=Path, default=settings.data_file)
    args = parser.parse_args()

    if not args.data_file.exists():
        sys.exit(f"Data file not found: {args.data_file.resolve()}\n"
                 f"Copy the Excel export into data/raw/ first (see README).")

    t0 = time.time()
    print(f"Loading + cleaning {args.data_file.name} ...")
    df = load_clean_events(args.data_file, limit=args.limit,
                           max_wait_sec=settings.max_wait_sec)
    print(f"  {len(df):,} clean events "
          f"({df['wait_start'].min():%Y-%m-%d} -> {df['wait_start'].max():%Y-%m-%d}), "
          f"median wait {df['wait_min'].median():.1f} min")

    print(f"Loading embedding model {settings.embedding_model} ...")
    model = get_model()

    client = get_client()
    ensure_collection(client, recreate=args.recreate)

    total = 0
    for start in tqdm(range(0, len(df), EMBED_CHUNK), desc="Embed + upsert", unit="chunk"):
        chunk = df.iloc[start : start + EMBED_CHUNK]
        sentences = [event_sentence(row) for _, row in chunk.iterrows()]
        vectors = model.encode(sentences, batch_size=256,
                               normalize_embeddings=True).tolist()
        total += upsert_events(client, chunk, vectors)

    status = collection_status(client)
    print(f"\nDone in {time.time() - t0:,.0f}s — upserted {total:,} events, "
          f"collection now holds {status['points']:,} points.")
    print("Verify:  python scripts/query_similar.py --service-type Consultation --hour 9 --dow 3")


if __name__ == "__main__":
    main()
