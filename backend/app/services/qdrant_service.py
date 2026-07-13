"""Qdrant collection management, upsert, and filtered nearest-event search."""

import uuid

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from app.config import settings

_NAMESPACE = uuid.UUID("7a1d8f6e-3b2c-4e5f-9a0b-1c2d3e4f5a6b")


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    name = settings.qdrant_collection
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )
        for field, schema in [
            ("clinic", PayloadSchemaType.KEYWORD),
            ("service_type", PayloadSchemaType.KEYWORD),
            ("service_point", PayloadSchemaType.KEYWORD),
            ("hour", PayloadSchemaType.INTEGER),
            ("day_of_week", PayloadSchemaType.INTEGER),
            ("is_weekend", PayloadSchemaType.INTEGER),
            ("wait_min", PayloadSchemaType.FLOAT),
        ]:
            client.create_payload_index(name, field_name=field, field_schema=schema)


def point_id(row: pd.Series) -> str:
    """Deterministic ID → idempotent re-ingestion (spec §3.1)."""
    key = f"{row['Anonymized_ID']}|{row['wait_start'].isoformat()}|{row['service_point_norm']}"
    return str(uuid.uuid5(_NAMESPACE, key))


def row_payload(row: pd.Series) -> dict:
    return {
        "patient_id": int(row["Anonymized_ID"]),
        "clinic": row["Clinic Name"],
        "service_type": row["Service Type"],
        "service_point": row["service_point_norm"],
        "hour": int(row["hour"]),
        "day_of_week": int(row["day_of_week"]),
        "is_weekend": int(row["is_weekend"]),
        "is_public_holiday": int(row["is_public_holiday"]),
        "month": int(row["month"]),
        "wait_min": float(row["wait_min"]),
        "contact_min": float(row["contact_min"]),
        "wait_start_iso": row["wait_start"].isoformat(),
    }


def upsert_events(client: QdrantClient, df: pd.DataFrame,
                  vectors: list[list[float]], batch_size: int = 512) -> int:
    points = [
        PointStruct(id=point_id(row), vector=vec, payload=row_payload(row))
        for (_, row), vec in zip(df.iterrows(), vectors)
    ]
    for i in range(0, len(points), batch_size):
        client.upsert(settings.qdrant_collection, points[i : i + batch_size], wait=True)
    return len(points)


def build_filter(clinic: str | None = None, service_type: str | None = None,
                 hour: int | None = None, hour_window: int = 2,
                 day_of_week: int | None = None,
                 weekend_class: bool = False) -> Filter | None:
    """Operational pre-filter for hybrid retrieval (spec §3.1)."""
    must: list[FieldCondition] = []
    if clinic:
        must.append(FieldCondition(key="clinic", match=MatchValue(value=clinic)))
    if service_type:
        must.append(FieldCondition(key="service_type", match=MatchValue(value=service_type)))
    if hour is not None:
        must.append(FieldCondition(
            key="hour", range=Range(gte=max(0, hour - hour_window), lte=min(23, hour + hour_window))
        ))
    if day_of_week is not None:
        if weekend_class:
            must.append(FieldCondition(
                key="is_weekend", match=MatchValue(value=1 if day_of_week >= 5 else 0)
            ))
        else:
            must.append(FieldCondition(key="day_of_week", match=MatchValue(value=day_of_week)))
    return Filter(must=must) if must else None


def search_similar(client: QdrantClient, query_vector: list[float],
                   query_filter: Filter | None, limit: int = 50) -> list[dict]:
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    ).points
    return [{"score": h.score, **h.payload} for h in hits]


def collection_status(client: QdrantClient) -> dict:
    name = settings.qdrant_collection
    if not client.collection_exists(name):
        return {"exists": False, "points": 0}
    info = client.get_collection(name)
    return {"exists": True, "points": info.points_count, "status": str(info.status)}
