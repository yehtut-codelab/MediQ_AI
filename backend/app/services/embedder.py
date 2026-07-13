"""Sentence-transformers embedding wrapper (local, no API cost)."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str], batch_size: int = 256) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
    )
    return vectors.tolist()


def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]
