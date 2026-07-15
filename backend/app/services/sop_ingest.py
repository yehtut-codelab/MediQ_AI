"""Shared SOP document extraction + chunk/embed/upsert logic — used by both
scripts/ingest_sop_docs.py (bulk folder ingestion) and the
/api/v1/documents upload/reingest endpoints (single-file ingestion via the UI).
"""

from pathlib import Path

from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient

from app.config import settings
from app.services.embedder import embed_texts
from app.services.sop_service import upsert_chunks

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def extract_text(path: Path) -> str:
    if path.suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if path.suffix == ".docx":
        return "\n".join(p.text for p in DocxDocument(str(path)).paragraphs)
    return path.read_text(encoding="utf-8", errors="ignore")


def ingest_file(client: QdrantClient, document_id: str, document_name: str, path: Path) -> int:
    """Extract, chunk, embed, and upsert one file. Returns chunk count.

    Raises ValueError if the file has no extractable text.
    """
    text = extract_text(path)
    if not text.strip():
        raise ValueError(f"No extractable text in {path.name}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.sop_chunk_size, chunk_overlap=settings.sop_chunk_overlap,
    )
    pieces = splitter.split_text(text)
    chunks = [
        {"document_id": document_id, "document_name": document_name, "chunk_index": i, "text": piece}
        for i, piece in enumerate(pieces)
    ]
    vectors = embed_texts([c["text"] for c in chunks])
    return upsert_chunks(client, chunks, vectors)
