import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from qdrant_client import models

from app.agents.graph import wait_estimate_graph
from app.agents.qa_graph import qa_graph
from app.agents.supervisor_graph import supervisor_graph
from app.config import settings
from app.schemas import (
    ArrivalRequest,
    DocumentOut,
    EstimateResponse,
    ForecastBucket,
    PatientRegistration,
    QueueSimulationRequest,
    QueueSimulationResponse,
    RegisteredPatient,
    SamplePatient,
    SimilarEvent,
)
from app.services import document_registry, livestate, models_service, registry, stations
from app.services.pathway import build_pathway, classify
from app.services.qdrant_service import collection_status, get_client
from app.services.sop_ingest import SUPPORTED_EXTENSIONS, ingest_file
from app.services.sop_service import delete_document, ensure_sop_collection
from app.services.triage import triage

router = APIRouter(prefix="/api/v1")

SGT_BANDS = [(30, "green"), (60, "amber")]  # else red — matches digital-twin thresholds


def band(wait_min: float) -> str:
    for limit, name in SGT_BANDS:
        if wait_min < limit:
            return name
    return "red"


# RAG service_type (Qdrant label) -> trained-model category (Clean_Dataset label).
# The two datasets were built from different exports with different label
# schemes; "other" is a legitimate trained-model bucket, so unmapped service
# types still get a model prediction rather than none at all.
SERVICE_TYPE_TO_CATEGORY: dict[str, str] = {
    "Consultation": "consultation",
    "OCT": "diagnostic_scan",
    "Diagnostic": "diagnostic_scan",
    "VA": "visual_acuity",
    "HVF": "visual_field_test",
    "Biometry": "biometry",
    "Treatment": "treatment",
    "Orthoptic": "other",
    "Financial Con": "other",
    "PAT": "other",
    "Pre Consultation Test": "other",
}


def _run_queue_models(clinic: str, category: str, at: datetime, *,
                      visit_position: int | None = None,
                      visit_length: int | None = None,
                      appointment_lag_min: float = 0.0) -> dict:
    """XGBoost wait estimate + HMM congestion state + LSTM queue-depth
    forecast for one clinic/category at a tz-naive timestamp inside
    livestate.dataset_range(). Shared by /queue/simulate and /patients/register
    so both surfaces run the exact same model pipeline."""
    profile = livestate.typical_visit_profile(clinic, category)
    vp = visit_position if visit_position is not None else profile["visit_position"]
    vl = visit_length if visit_length is not None else profile["visit_length"]

    depth = livestate.queue_depth(clinic, category, at)
    model_wait = models_service.predict_wait_min(
        category=category, clinic=clinic, hour=at.hour, day_of_week=at.weekday(),
        month=at.month, visit_position=vp, visit_length=vl,
        appointment_lag_min=appointment_lag_min, queue_depth=depth,
    )
    mae = models_service.wait_meta().get("evaluation", {}).get("val_MAE_min", 0.0)

    waits = livestate.recent_waits(clinic, category, at)
    congestion = models_service.congestion_state(category, waits)

    feats = livestate.lstm_bucket_features(clinic, category, at)
    bucket_min = models_service.lstm_meta()["bucket_minutes"]
    if feats is None:
        forecast_vals: list[float] = []
        trend = "no-data"
    else:
        forecast_vals = models_service.forecast_queue_depth(feats)
        current = float(feats[-1][0])
        if not forecast_vals:
            trend = "no-data"
        elif forecast_vals[0] > current * 1.15:
            trend = "rising"
        elif forecast_vals[0] < current * 0.85:
            trend = "falling"
        else:
            trend = "stable"

    return {
        "queue_depth": depth,
        "model_estimate_min": round(model_wait, 1),
        "model_mae_min": mae,
        "congestion_state": congestion["state"],
        "congestion_mean_wait_min": congestion.get("state_mean_wait"),
        "congestion_observations": congestion.get("n_observations", 0),
        "forecast": [
            ForecastBucket(minutes_ahead=bucket_min * (i + 1), queue_depth=v)
            for i, v in enumerate(forecast_vals)
        ],
        "forecast_trend": trend,
    }


@router.post("/arrivals/estimate", response_model=EstimateResponse)
def estimate_arrival(req: ArrivalRequest) -> EstimateResponse:
    local = req.arrival_datetime
    result = wait_estimate_graph.invoke({
        "clinic": req.clinic,
        "service_type": req.service_type,
        "hour": local.hour,
        "day_of_week": local.weekday(),
        "diagnosis": req.diagnosis,
        "medical_history": req.medical_history,
        "current_issue": req.current_issue,
    })
    if not result.get("hits"):
        raise HTTPException(404, "No historical evidence found — has ingestion run?")

    stats = result["stats"]
    return EstimateResponse(
        estimated_wait_min=stats["median"],
        range_min=(stats["p10"], stats["p90"]),
        confidence=result["confidence"],
        evidence_count=stats["n"],
        filters_widened=result["widenings"],
        similar_events_sample=[
            SimilarEvent(**{k: h.get(k) for k in SimilarEvent.model_fields})
            for h in result["hits"][:10]
        ],
        explanation=result["explanation"],
        status_band=band(stats["median"]),
    )


@router.get("/arrivals/samples", response_model=list[SamplePatient])
def sample_arrivals(n: int = 5) -> list[SamplePatient]:
    n = max(1, min(n, 20))
    points = get_client().query_points(
        collection_name=settings.qdrant_collection,
        query=models.SampleQuery(sample=models.Sample.RANDOM),
        limit=n,
        with_payload=True,
    ).points
    if not points:
        raise HTTPException(404, "No events in collection — has ingestion run?")
    return [
        SamplePatient(
            patient_id=p.payload["patient_id"],
            clinic=p.payload["clinic"],
            service_type=p.payload["service_type"],
            service_point=p.payload["service_point"],
            arrival_iso=p.payload["wait_start_iso"],
            actual_wait_min=p.payload["wait_min"],
        )
        for p in points
    ]


def _replay_note(requested: datetime, replayed_at: datetime) -> str:
    lo, hi = livestate.dataset_range()
    return (
        f"{requested.date()} is outside the trained-model dataset range "
        f"({lo.date()} … {hi.date()}); showing a pattern replay from "
        f"{replayed_at.strftime('%a %Y-%m-%d %H:%M')} — the nearest date with "
        f"the same weekday and time-of-day."
    )


@router.post("/queue/simulate", response_model=QueueSimulationResponse)
def simulate_queue(req: QueueSimulationRequest) -> QueueSimulationResponse:
    """Model-based queue simulation: XGBoost wait estimate + HMM congestion
    state + LSTM queue-depth forecast, driven by the trained models in
    Models/ rather than RAG retrieval. Clinic state is replayed from the
    trained-model dataset (Clean_Dataset); a timestamp outside that dataset's
    range is mapped to the nearest analog (same weekday/time-of-day) instead
    of being rejected — see livestate.analog_timestamp."""
    if req.category not in livestate.CATEGORIES:
        raise HTTPException(422, f"category must be one of {livestate.CATEGORIES}")

    requested = req.as_of.replace(tzinfo=None)
    replayed_at, is_analog = livestate.analog_timestamp(requested)

    m = _run_queue_models(
        req.clinic, req.category, replayed_at,
        visit_position=req.visit_position,
        visit_length=req.visit_length,
        appointment_lag_min=req.appointment_lag_min,
    )
    return QueueSimulationResponse(
        clinic=req.clinic,
        category=req.category,
        as_of=requested,
        replayed_at=replayed_at,
        is_analog_replay=is_analog,
        replay_note=_replay_note(requested, replayed_at) if is_analog else None,
        queue_depth=m["queue_depth"],
        model_estimate_min=m["model_estimate_min"],
        model_mae_min=m["model_mae_min"],
        status_band=band(m["model_estimate_min"]),
        congestion_state=m["congestion_state"],
        congestion_mean_wait_min=m["congestion_mean_wait_min"],
        congestion_observations=m["congestion_observations"],
        forecast=m["forecast"],
        forecast_trend=m["forecast_trend"],
    )


@router.post("/patients/register", response_model=RegisteredPatient)
def register_patient(req: PatientRegistration) -> RegisteredPatient:
    local = req.arrival_datetime
    result = wait_estimate_graph.invoke({
        "clinic": req.clinic,
        "service_type": req.service_type,
        "hour": local.hour,
        "day_of_week": local.weekday(),
        "diagnosis": req.diagnosis,
        "medical_history": req.medical_history,
        "current_issue": req.current_issue,
    })
    if not result.get("hits"):
        raise HTTPException(404, "No historical evidence found — has ingestion run?")

    stats = result["stats"]
    queue_position = registry.next_queue_position(
        req.clinic, req.service_type, local.date().isoformat()
    )
    triage_result = triage(req.diagnosis, req.medical_history, req.current_issue)
    archetype = classify(req.diagnosis, req.medical_history, req.current_issue,
                         priority=triage_result.priority)
    pathway = build_pathway(archetype)

    # Model-based queue state (XGBoost/HMM/LSTM), same pipeline as
    # /queue/simulate. A timestamp outside the trained-model dataset's range
    # is replayed from the nearest same-weekday/time-of-day analog instead of
    # being skipped — the RAG estimate above always applies regardless, since
    # it only reads hour-of-day/weekday off the timestamp, not the actual
    # calendar date.
    naive_arrival = local.replace(tzinfo=None) if local.tzinfo else local
    model_category = SERVICE_TYPE_TO_CATEGORY.get(req.service_type, "other")
    replayed_at, is_analog = livestate.analog_timestamp(naive_arrival)
    model_fields: dict = {}
    try:
        m = _run_queue_models(req.clinic, model_category, replayed_at)
        model_fields = {
            "model_category": model_category,
            "model_estimate_min": m["model_estimate_min"],
            "model_mae_min": m["model_mae_min"],
            "model_queue_depth": m["queue_depth"],
            "model_congestion_state": m["congestion_state"],
            "model_congestion_mean_wait_min": m["congestion_mean_wait_min"],
            "model_forecast": m["forecast"],
            "model_forecast_trend": m["forecast_trend"],
            "model_replayed_at": replayed_at,
            "model_is_analog_replay": is_analog,
            "model_replay_note": _replay_note(naive_arrival, replayed_at) if is_analog else None,
        }
    except Exception:
        model_fields = {"model_unavailable_reason": "Model prediction failed unexpectedly."}

    patient = RegisteredPatient(
        patient_id=req.patient_id or random.randint(300_000_000, 399_999_999),
        display_name=req.display_name,
        clinic=req.clinic,
        service_type=req.service_type,
        diagnosis=req.diagnosis,
        medical_history=req.medical_history,
        current_issue=req.current_issue,
        arrival_datetime=req.arrival_datetime,
        appointment_datetime=req.appointment_datetime,
        queue_position=queue_position,
        priority=triage_result.priority,
        priority_label=triage_result.label,
        triage_reasons=triage_result.reasons,
        pathway_label=pathway["label"],
        pathway=pathway["steps"],
        total_visit_min=pathway["total_visit_min"],
        estimated_wait_min=stats["median"],
        range_min=(stats["p10"], stats["p90"]),
        status_band=band(stats["median"]),
        explanation=result["explanation"],
        registered_at=datetime.now(timezone.utc),
        **model_fields,
    )
    registry.add(patient)
    return patient


@router.get("/patients", response_model=list[RegisteredPatient])
def list_patients() -> list[RegisteredPatient]:
    return registry.list_all()


@router.get("/stations/overview")
def stations_overview() -> dict:
    """Per-station historical stats + today's registered counts, grouped by clinic."""
    overview = stations.station_overview(get_client())
    today = datetime.now(timezone.utc).astimezone().date().isoformat()
    counts = registry.today_counts(today)
    return {
        clinic: [
            {**s, "registered_today": counts.get((clinic, s["service_type"]), 0)}
            for s in station_list
        ]
        for clinic, station_list in overview.items()
    }


class AgentAnalyzeRequest(BaseModel):
    clinic: str = Field(..., examples=["TTSH Eye Centre", "Clinic 1A"])
    category: str = Field(..., examples=["consultation", "visual_acuity", "diagnostic_scan"])
    as_of: datetime = Field(..., description="Timestamp within the cleaned dataset range")
    objective: str = "Analyse bottlenecks and suggest resource optimization"


@router.post("/agent/analyze")
def agent_analyze(req: AgentAnalyzeRequest) -> dict:
    """Run the full supervisor multi-agent analysis (Figure 2 architecture).
    A requested `as_of` outside the trained-model dataset's range is replayed
    from the nearest same-weekday/time-of-day analog instead of being
    rejected — see livestate.analog_timestamp."""
    requested = req.as_of.replace(tzinfo=None)
    replayed_at, is_analog = livestate.analog_timestamp(requested)
    result = supervisor_graph.invoke({
        "clinic": req.clinic,
        "category": req.category,
        "as_of": replayed_at,
        "objective": req.objective,
    })
    report = result["report"]
    report["requested_as_of"] = requested.isoformat()
    report["is_analog_replay"] = is_analog
    report["replay_note"] = _replay_note(requested, replayed_at) if is_analog else None
    return report


@router.get("/agent/dataset-range")
def agent_dataset_range() -> dict:
    lo, hi = livestate.dataset_range()
    return {"min": lo.isoformat(), "max": hi.isoformat(),
            "categories": livestate.CATEGORIES}


class QARequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


@router.post("/qa/ask")
def qa_ask(req: QARequest) -> dict:
    """SOP & Healthcare Q&A: a tool-calling LangGraph agent searches the
    `mediq_sop_docs` Qdrant collection and answers grounded in what it finds.
    Unlike the other graphs, there is no template fallback here — open-ended
    reasoning over retrieved text requires an LLM."""
    if not settings.openai_api_key:
        raise HTTPException(503, "SOP Q&A requires OPENAI_API_KEY to be configured.")

    from langchain_core.messages import HumanMessage

    result = qa_graph.invoke({"messages": [HumanMessage(req.question)], "citations": []})

    seen: set[tuple[str, str]] = set()
    sources = []
    for c in result.get("citations", []):
        key = (c["title"], c["snippet"])
        if key not in seen:
            seen.add(key)
            sources.append(c)

    return {
        "status": "ok",
        "question": req.question,
        "answer": result["messages"][-1].content,
        "sources": sources,
    }


def _document_out(row) -> DocumentOut:
    return DocumentOut(
        id=row["id"], name=row["name"], ext=row["ext"], size_bytes=row["size_bytes"],
        status=row["status"], chunk_count=row["chunk_count"] or 0,
        error_message=row["error_message"], uploaded_at=row["uploaded_at"],
        ingested_at=row["ingested_at"],
    )


@router.get("/documents", response_model=list[DocumentOut])
def list_documents() -> list[DocumentOut]:
    """Documents in the SOP knowledge base, newest first — powers the document
    management page (upload/status/remove)."""
    return [_document_out(r) for r in document_registry.list_all()]


@router.post("/documents/upload", response_model=DocumentOut, status_code=201)
def upload_document(file: UploadFile = File(...)) -> DocumentOut:
    """Upload one SOP document (PDF/DOCX/TXT) and ingest it immediately into
    the `mediq_sop_docs` Qdrant collection."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: pdf, docx, txt")

    settings.sop_docs_dir.mkdir(parents=True, exist_ok=True)
    document_id = str(uuid.uuid4())
    dest = settings.sop_docs_dir / f"{document_id}{ext}"
    contents = file.file.read()
    dest.write_bytes(contents)

    document_registry.add(document_id, file.filename or dest.name, str(dest.resolve()),
                          ext, len(contents), status="processing")

    try:
        client = get_client()
        ensure_sop_collection(client)
        n = ingest_file(client, document_id, Path(file.filename or dest.name).stem, dest)
        document_registry.update_status(document_id, "ingested", chunk_count=n)
    except Exception as exc:
        document_registry.update_status(document_id, "failed", error_message=str(exc))

    return _document_out(document_registry.get(document_id))


@router.post("/documents/{document_id}/reingest", response_model=DocumentOut)
def reingest_document(document_id: str) -> DocumentOut:
    """Retry ingestion for a document stuck in `failed` (or refresh an existing one)."""
    row = document_registry.get(document_id)
    if not row:
        raise HTTPException(404, "Document not found")

    path = Path(row["file_path"])
    if not path.exists():
        document_registry.update_status(document_id, "failed",
                                        error_message="Source file no longer on disk.")
        raise HTTPException(404, "Source file no longer on disk — re-upload instead.")

    try:
        client = get_client()
        ensure_sop_collection(client)
        delete_document(client, document_id)
        n = ingest_file(client, document_id, path.stem, path)
        document_registry.update_status(document_id, "ingested", chunk_count=n)
    except Exception as exc:
        document_registry.update_status(document_id, "failed", error_message=str(exc))

    return _document_out(document_registry.get(document_id))


@router.delete("/documents/{document_id}", status_code=204)
def delete_document_endpoint(document_id: str) -> None:
    """Remove a document's chunks from Qdrant, its file on disk, and its registry row."""
    row = document_registry.get(document_id)
    if not row:
        raise HTTPException(404, "Document not found")

    delete_document(get_client(), document_id)
    path = Path(row["file_path"])
    if path.exists():
        path.unlink()
    document_registry.delete(document_id)


@router.get("/collection/status")
def qdrant_status() -> dict:
    return collection_status(get_client())
