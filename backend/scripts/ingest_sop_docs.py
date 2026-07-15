"""Ingest SOP / healthcare-procedure documents (PDF, DOCX, TXT) into Qdrant.

Documents ingested here are also registered in the same SQLite registry used
by the document management UI (POST /api/v1/documents/upload), so they show
up in the management page regardless of how they were added.

Usage:
    python scripts/ingest_sop_docs.py                  # ingest data/sop_docs/
    python scripts/ingest_sop_docs.py --dir some/path
    python scripts/ingest_sop_docs.py --recreate        # drop & rebuild collection
"""

import argparse
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from app.config import settings
from app.services import document_registry
from app.services.qdrant_service import get_client
from app.services.sop_ingest import SUPPORTED_EXTENSIONS, ingest_file
from app.services.sop_service import ensure_sop_collection, sop_collection_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=settings.sop_docs_dir)
    parser.add_argument("--recreate", action="store_true", help="Drop and rebuild collection")
    args = parser.parse_args()

    if not args.dir.exists():
        sys.exit(f"SOP docs directory not found: {args.dir.resolve()}\n"
                 f"Create it and add PDF/DOCX/TXT files (see README).")

    files = [p for p in sorted(args.dir.rglob("*")) if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        sys.exit(f"No PDF/DOCX/TXT files found in {args.dir.resolve()}")

    client = get_client()
    ensure_sop_collection(client, recreate=args.recreate)

    t0 = time.time()
    total_chunks = 0
    for path in tqdm(files, desc="Ingest documents", unit="doc"):
        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())))
        document_registry.add(document_id, path.name, str(path.resolve()),
                              path.suffix.lower(), path.stat().st_size, status="processing")
        try:
            n = ingest_file(client, document_id, path.stem, path)
            document_registry.update_status(document_id, "ingested", chunk_count=n)
            total_chunks += n
        except Exception as exc:
            document_registry.update_status(document_id, "failed", error_message=str(exc))
            print(f"  failed: {path.name} — {exc}")

    status = sop_collection_status(client)
    print(f"\nDone in {time.time() - t0:,.0f}s — upserted {total_chunks:,} chunks from "
          f"{len(files)} documents, collection now holds {status['points']:,} points.")


if __name__ == "__main__":
    main()
