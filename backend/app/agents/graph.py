"""LangGraph WaitEstimateGraph (spec §3.2, ADR-001).

parse_context -> retrieve -> check_evidence -(widen, max 2)-> retrieve
                                 |
                                 v
                     rerank -> compute_stats -> synthesize -> END

Hybrid retrieval: the query embeds the patient's clinical presentation
(diagnosis, history, current issue) plus operational context; Qdrant applies
operational pre-filters over the vector search; a cross-encoder then reranks
the candidate pool and the estimate is generated from the top 10 cases.
"""

import statistics
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.config import settings
from app.services.embedder import embed_one
from app.services.preprocess import DAY_NAMES, daypart
from app.services.qdrant_service import build_filter, get_client, search_similar
from app.services.reranker import rerank

MIN_EVIDENCE = 20
K = 50
TOP_N = 10


class EstimateState(TypedDict, total=False):
    # inputs
    clinic: str
    service_type: str
    hour: int
    day_of_week: int
    diagnosis: str | None
    medical_history: str | None
    current_issue: str | None
    # working
    query_text: str
    query_vector: list[float]
    widenings: int
    retrieved_n: int
    hits: list[dict[str, Any]]
    # outputs
    stats: dict[str, float]
    explanation: str
    confidence: str


def clinical_narrative(state: EstimateState) -> str:
    parts = []
    if state.get("diagnosis"):
        parts.append(f"Diagnosis: {state['diagnosis']}.")
    if state.get("medical_history"):
        parts.append(f"Medical history: {state['medical_history']}.")
    if state.get("current_issue"):
        parts.append(f"Current issue: {state['current_issue']}.")
    return " ".join(parts)


def parse_context(state: EstimateState) -> EstimateState:
    query = (
        f"{state['service_type']} at {state['clinic']}, "
        f"on {DAY_NAMES[state['day_of_week']]} at {state['hour']:02d}:00, "
        f"{daypart(state['hour'])} "
        f"{'weekend' if state['day_of_week'] >= 5 else 'weekday'}."
    )
    narrative = clinical_narrative(state)
    if narrative:
        query = f"{query} {narrative}"
    return {"query_text": query, "query_vector": embed_one(query), "widenings": 0}


def retrieve(state: EstimateState) -> EstimateState:
    w = state["widenings"]
    # widening ladder: 0 = exact dow + hour±2; 1 = weekend-class + hour±2; 2 = no time filter
    qfilter = build_filter(
        clinic=state["clinic"],
        service_type=state["service_type"],
        hour=state["hour"] if w < 2 else None,
        day_of_week=state["day_of_week"] if w < 2 else None,
        weekend_class=(w == 1),
    )
    hits = search_similar(get_client(), state["query_vector"], qfilter, limit=K)
    return {"hits": hits}


def check_evidence(state: EstimateState) -> str:
    if len(state["hits"]) >= MIN_EVIDENCE or state["widenings"] >= 2:
        return "enough"
    return "widen"


def widen(state: EstimateState) -> EstimateState:
    return {"widenings": state["widenings"] + 1}


def rerank_hits(state: EstimateState) -> EstimateState:
    """Cross-encoder rerank of the retrieved pool; keep the top-N cases."""
    top = rerank(state["query_text"], state["hits"], top_n=TOP_N)
    return {"hits": top, "retrieved_n": len(state["hits"])}


def compute_stats(state: EstimateState) -> EstimateState:
    waits = sorted(h["wait_min"] for h in state["hits"]) or [0.0]
    n = len(waits)
    p = lambda q: waits[min(n - 1, int(q * n))]
    stats = {
        "median": round(statistics.median(waits), 1),
        "p10": round(p(0.10), 1),
        "p75": round(p(0.75), 1),
        "p90": round(p(0.90), 1),
        "n": n,
    }
    pool = state.get("retrieved_n", n)
    confidence = "high" if pool >= 40 and state["widenings"] == 0 else \
                 "medium" if pool >= MIN_EVIDENCE else "low"
    return {"stats": stats, "confidence": confidence}


def synthesize(state: EstimateState) -> EstimateState:
    """LLM explanation grounded in code-computed stats; template fallback without a key."""
    s = state["stats"]
    dow = DAY_NAMES[state["day_of_week"]]
    pool = state.get("retrieved_n", s["n"])
    base = (
        f"Based on the top {s['n']} most similar historical cases "
        f"(reranked from {pool} retrieved {dow} {daypart(state['hour'])} "
        f"{state['service_type']} visits at {state['clinic']}), the typical wait is "
        f"about {s['median']:.0f} minutes (90% of these patients waited under "
        f"{s['p90']:.0f} minutes)."
    )
    narrative = clinical_narrative(state)
    if narrative:
        base += f" Patient presentation considered in matching — {narrative}"
    if state["widenings"]:
        base += f" Note: filters were widened {state['widenings']}x due to sparse exact matches."

    if not settings.anthropic_api_key:
        return {"explanation": base}

    try:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=settings.llm_model, max_tokens=300,
                            api_key=settings.anthropic_api_key)
        msg = llm.invoke(
            "Rewrite this hospital wait estimate as one short, warm, patient-friendly "
            "message followed by one staff-facing sentence. Keep every number exactly "
            f"as given, do not invent numbers:\n\n{base}"
        )
        return {"explanation": msg.content}
    except Exception:
        return {"explanation": base}  # LLM is optional (NFR-4)


def build_graph():
    g = StateGraph(EstimateState)
    g.add_node("parse_context", parse_context)
    g.add_node("retrieve", retrieve)
    g.add_node("widen", widen)
    g.add_node("rerank", rerank_hits)
    g.add_node("compute_stats", compute_stats)
    g.add_node("synthesize", synthesize)

    g.set_entry_point("parse_context")
    g.add_edge("parse_context", "retrieve")
    g.add_conditional_edges("retrieve", check_evidence,
                            {"enough": "rerank", "widen": "widen"})
    g.add_edge("widen", "retrieve")
    g.add_edge("rerank", "compute_stats")
    g.add_edge("compute_stats", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


wait_estimate_graph = build_graph()
