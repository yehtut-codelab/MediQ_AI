"""OpenAI embedding wrapper — same provider as the LLM, so wait-time RAG and
the SOP Q&A agent share one API key/provider (app/config.py: OPENAI_API_KEY)."""

from functools import lru_cache

from openai import OpenAI

from app.config import settings

EMBED_BATCH = 2048  # OpenAI embeddings API accepts up to 2048 inputs per request


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required — embeddings are computed via the OpenAI API "
            "(app/services/embedder.py). Set it in backend/.env."
        )
    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH) -> list[list[float]]:
    client = get_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        resp = client.embeddings.create(model=settings.embedding_model, input=chunk)
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]
