# TTSH Eye Clinic — Agentic Queue & Wait Time Advisor

NUS-ISS Capstone (Team 2). RAG system over historical TTSH Eye Centre wait time events:
new patient arrivals are matched against nearest historical events in **Qdrant**, and a
**LangGraph** agent produces an evidence-based wait/queue estimate.

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

Copy `TTSH Oct 25 - 04 May 26 - WaitTimeAdded.xlsx` into `data/raw/` (gitignored).

### 4. Ingest

```bash
python scripts/ingest_waittime.py                 # full run (~237K events)
python scripts/ingest_waittime.py --limit 5000    # smoke test
python scripts/ingest_waittime.py --recreate      # drop & rebuild collection
```

### 5. Test nearest-event search

```bash
python scripts/query_similar.py --clinic "TTSH Eye Centre" --service-type Consultation --hour 9 --dow 3
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
QDRANT_COLLECTION=ttsh_wait_events
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ANTHROPIC_API_KEY=            # optional — template fallback without it
```
"# MediQ_AI" 
"# MediQ_AI" 
"# MediQ_AI" 
