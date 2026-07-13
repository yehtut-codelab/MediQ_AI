# MediQ AI — Agentic Digital Twin for Eye Clinic Operations

MediQ AI models an eye clinic as a **digital twin**: historical patient-flow data drives a
live simulation of queue depth, congestion, and wait times for any clinic/category/timestamp,
and a set of **LangGraph** agents orchestrate trained ML models and retrieval-augmented
evidence into wait estimates, bottleneck alerts, and operational recommendations.

📄 Docs: [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) · [`docs/FRAMEWORK_DECISION.md`](docs/FRAMEWORK_DECISION.md)

## Architecture

**Digital twin layer** (`services/livestate.py`) replays the historical dataset as live
clinic state — queue depth, recent wait times, and bucketed features — for any requested
timestamp. A timestamp outside the trained dataset's date range is mapped to the nearest
same-weekday/time-of-day analog rather than rejected, so the twin always has something to
simulate against.

**Trained models** (`Models/`) provide the quantitative core:
- XGBoost regressor — wait-time estimate
- Per-category Gaussian HMM — congestion state (Low/Medium/High)
- LSTM — 15-minute-bucket queue-depth forecast

**RAG evidence layer** (Qdrant + cross-encoder reranker) retrieves the nearest historical
visit events for a new arrival, giving every estimate a grounded, inspectable evidence set
alongside the model prediction.

**Agentic orchestration** — two LangGraph graphs:

- **`WaitEstimateGraph`** (`agents/graph.py`) — single-arrival RAG estimate:
  `parse_context → retrieve → check_evidence →(widen, up to 2×)→ rerank → compute_stats → synthesize`
  Retrieval filters widen automatically when too few comparable historical cases are found.

- **Supervisor multi-agent graph** (`agents/supervisor_graph.py`) — clinic-wide operational
  analysis, fanning out to four specialists in parallel and fusing their findings:

  ```
                      ┌─ waiting_time_agent ──┐
  supervisor ─ fan-out┼─ bottleneck_agent ────┼─ result_fusion ─ critique
                      ├─ resource_planning ───┤        │
                      └─ predictive_agent ────┘        ▼
                                         decision_support ─ reporting ─ END
  ```

  Each specialist reads live twin state and calls a trained model; `critique` validates
  cross-model agreement (e.g. XGBoost vs. RAG spread, HMM-vs-forecast consistency) and scores
  confidence before `decision_support` produces prioritized recommendations.

Both graphs optionally call Claude to phrase the final explanation/report in plain language —
every number in that text is computed upstream by code/models, and both graphs fall back to a
template summary if no API key is configured (the system runs fully without an LLM).

## Quick Start — Phase 1 (Excel → Qdrant → nearest-event search)

### 1. Start Qdrant

```bash
docker compose up -d qdrant
# dashboard: http://localhost:6333/dashboard
```

### 2. Install backend deps

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Place the dataset

Copy the wait time dataset Excel file into `data/raw/` (gitignored).

### 4. Ingest

```bash
python scripts/ingest_waittime.py                 # full run (~237K events)
python scripts/ingest_waittime.py --limit 5000    # smoke test
python scripts/ingest_waittime.py --recreate      # drop & rebuild collection
```

### 5. Test nearest-event search

```bash
python scripts/query_similar.py --clinic "Eye Centre" --service-type Consultation --hour 9 --dow 3
```

### 6. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

## Key endpoints

| Endpoint | Graph / model | Purpose |
|---|---|---|
| `POST /api/v1/arrivals/estimate` | `WaitEstimateGraph` | RAG wait estimate for a new arrival |
| `POST /api/v1/patients/register` | `WaitEstimateGraph` + trained models | Register a patient, triage, pathway, wait estimate |
| `POST /api/v1/queue/simulate` | trained models | Model-only queue simulation (twin replay) |
| `POST /api/v1/agent/analyze` | Supervisor multi-agent graph | Full clinic-wide bottleneck/resource/forecast analysis |
| `GET /api/v1/stations/overview` | — | Per-station historical stats + today's registrations |
| `POST /api/v1/qa/ask` | — | Placeholder for the SOP & Healthcare Q&A RAG (not yet connected) |

## Layout

```
backend/
  app/
    api/routes.py             # FastAPI endpoints
    agents/
      graph.py                 # LangGraph WaitEstimateGraph (RAG estimate)
      supervisor_graph.py       # LangGraph supervisor multi-agent graph (clinic analysis)
    services/
      livestate.py              # digital-twin state: queue depth, recent waits, forecasts
      models_service.py         # loads/serves XGBoost, HMM, LSTM
      embedder.py, qdrant_service.py, reranker.py   # RAG retrieval
      preprocess.py, pathway.py, triage.py, stations.py, registry.py
    config.py, schemas.py, main.py
  scripts/
    ingest_waittime.py    # Excel → Qdrant
    mine_pathways.py      # mine empirical patient pathways from wait time export
    query_similar.py      # CLI nearest-event check
Models/                  # trained model artifacts (XGBoost, HMM, LSTM + metadata)
frontend/                # Next.js dashboard — arrival simulator, registration, station map, agent console
docs/                    # spec + ADRs
data/raw/                # dataset (gitignored)
```

## Environment

`backend/.env` (copy from `.env.example`):

```
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=mediq_wait_events
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ANTHROPIC_API_KEY=            # optional — template fallback without it
```
