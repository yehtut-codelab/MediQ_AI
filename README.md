# MediQ AI — Digital Twin for Eye Clinic Operations

An agentic RAG system that models eye clinic operations as a digital twin: historical
wait time events are indexed in **Qdrant**, new patient arrivals are matched against the
nearest historical events, and a **LangGraph** agent produces an evidence-based wait/queue
estimate.

📄 Docs: [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) · [`docs/FRAMEWORK_DECISION.md`](docs/FRAMEWORK_DECISION.md)

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
# POST http://localhost:8000/api/v1/arrivals/estimate
```

## Layout

```
backend/
  app/
    api/routes.py        # FastAPI endpoints
    agents/graph.py      # LangGraph WaitEstimateGraph
    services/            # preprocess / embedder / qdrant / retrieval
    config.py, schemas.py, main.py
  scripts/
    ingest_waittime.py   # Excel → Qdrant
    query_similar.py     # CLI nearest-event check
frontend/                # Next.js dashboard (Phase 3)
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
