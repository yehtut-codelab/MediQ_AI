"""Supervisor/Orchestrator multi-agent graph — implements the Agentic AI Framework
from TTSH_AI_Agents_Architecture_Manuscript_Version.png (Figure 2).

                    ┌─ waiting_time_agent ──┐
supervisor ─ fan-out┼─ bottleneck_agent ────┼─ result_fusion ─ critique
                    ├─ resource_planning ───┤        │
                    └─ predictive_agent ────┘        ▼
                                       decision_support ─ reporting ─ END

Specialists run in parallel (LangGraph fan-out/fan-in). Trained models:
XGBoost (waiting time), per-category HMMs (bottlenecks), LSTM (queue forecast).
All numbers are computed in code; the optional LLM only phrases the report.
"""

import statistics
from datetime import datetime
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from app.config import settings
from app.services import livestate, models_service
from app.services.embedder import embed_one
from app.services.preprocess import DAY_NAMES, daypart
from app.services.qdrant_service import build_filter, get_client, search_similar

# cleaned-dataset categories → raw Qdrant service_type labels (for RAG evidence)
CATEGORY_TO_SERVICE_TYPE = {
    "consultation": "Consultation",
    "visual_acuity": "VA",
    "diagnostic_scan": "OCT",
    "visual_field_test": "HVF",
    "biometry": "Biometry",
    "treatment": "Treatment",
}


def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


class SupervisorState(TypedDict, total=False):
    # user request (parsed by supervisor)
    clinic: str
    category: str
    as_of: datetime
    objective: str
    # patient context for the wait estimate
    visit_position: int
    visit_length: int
    appointment_lag_min: float
    # specialist outputs merge here (parallel fan-in)
    findings: Annotated[dict[str, Any], _merge]
    # downstream modules
    fused: dict
    critique: dict
    decisions: list[dict]
    report: dict


# ── Supervisor: parse objective, prepare shared context ──────────────────

def supervisor(state: SupervisorState) -> SupervisorState:
    profile = livestate.typical_visit_profile(state["clinic"], state["category"])
    return {
        "visit_position": state.get("visit_position") or profile["visit_position"],
        "visit_length": state.get("visit_length") or profile["visit_length"],
        "appointment_lag_min": state.get("appointment_lag_min", 0.0),
        "findings": {},
    }


# ── Specialist 1: Waiting Time Analysis Agent (XGBoost + RAG evidence) ───

def waiting_time_agent(state: SupervisorState) -> SupervisorState:
    at, clinic, cat = state["as_of"], state["clinic"], state["category"]
    depth = livestate.queue_depth(clinic, cat, at)

    xgb_est = models_service.predict_wait_min(
        category=cat, clinic=clinic, hour=at.hour, day_of_week=at.weekday(),
        month=at.month, visit_position=state["visit_position"],
        visit_length=state["visit_length"],
        appointment_lag_min=state["appointment_lag_min"], queue_depth=depth,
    )

    rag = {"median": None, "n": 0}
    service_type = CATEGORY_TO_SERVICE_TYPE.get(cat)
    if service_type:
        query = (f"{service_type} at {clinic}, on {DAY_NAMES[at.weekday()]} "
                 f"at {at.hour:02d}:00, {daypart(at.hour)} weekday.")
        hits = search_similar(
            get_client(), embed_one(query),
            build_filter(clinic=clinic, service_type=service_type,
                         hour=at.hour, day_of_week=at.weekday()),
            limit=50,
        )
        if hits:
            rag = {"median": round(statistics.median(h["wait_min"] for h in hits), 1),
                   "n": len(hits)}

    return {"findings": {"waiting_time": {
        "model_estimate_min": round(xgb_est, 1),
        "model_mae_min": models_service.wait_meta()["evaluation"]["val_MAE_min"],
        "rag_median_min": rag["median"],
        "rag_evidence_n": rag["n"],
        "current_queue_depth": depth,
    }}}


# ── Specialist 2: Bottleneck Detection Agent (HMM congestion states) ─────

def bottleneck_agent(state: SupervisorState) -> SupervisorState:
    at, clinic = state["as_of"], state["clinic"]
    depths = livestate.queue_depths_all(clinic, at)
    stations = []
    for cat, depth in depths.items():
        waits = livestate.recent_waits(clinic, cat, at)
        hmm = models_service.congestion_state(cat, waits)
        stations.append({**hmm, "queue_depth": depth})
    bottlenecks = [s for s in stations
                   if s["state"] == "High" or (s["state"] == "Medium" and s["queue_depth"] >= 10)]
    return {"findings": {"bottlenecks": {
        "stations": stations,
        "flagged": bottlenecks,
    }}}


# ── Specialist 3: Resource Planning Agent (allocation options) ───────────

def resource_planning_agent(state: SupervisorState) -> SupervisorState:
    at, clinic = state["as_of"], state["clinic"]
    depths = livestate.queue_depths_all(clinic, at)
    total = sum(depths.values()) or 1
    options = []
    for cat, depth in sorted(depths.items(), key=lambda kv: -kv[1]):
        if depth < 5:
            continue
        share = depth / total
        options.append({
            "station": cat,
            "queue_depth": depth,
            "load_share_pct": round(share * 100, 1),
            "option": (
                f"Open an additional {cat.replace('_', ' ')} room/counter"
                if share > 0.35 else
                f"Reassign one floating staff member to {cat.replace('_', ' ')}"
            ),
        })
    return {"findings": {"resource_planning": {"options": options[:4]}}}


# ── Specialist 4: Predictive Analytics Agent (LSTM queue forecast) ───────

def predictive_agent(state: SupervisorState) -> SupervisorState:
    at, clinic, cat = state["as_of"], state["clinic"], state["category"]
    feats = livestate.lstm_bucket_features(clinic, cat, at)
    if feats is None:
        forecast, trend = [], "no-data"
    else:
        forecast = models_service.forecast_queue_depth(feats)
        current = float(feats[-1][0])
        trend = ("rising" if forecast and forecast[0] > current * 1.15 else
                 "falling" if forecast and forecast[0] < current * 0.85 else "stable")
    return {"findings": {"forecast": {
        "category": cat,
        "next_hour_queue_depth": forecast,   # 4 × 15-min buckets
        "trend": trend,
        "model_mae": models_service.lstm_meta()["evaluation"]["val_MAE"],
    }}}


# ── Result Fusion Module ─────────────────────────────────────────────────

def result_fusion(state: SupervisorState) -> SupervisorState:
    f = state["findings"]
    wt = f["waiting_time"]
    estimates = [wt["model_estimate_min"]]
    if wt["rag_median_min"] is not None:
        estimates.append(wt["rag_median_min"])
    return {"fused": {
        "consensus_wait_min": round(statistics.mean(estimates), 1),
        "estimate_spread_min": round(max(estimates) - min(estimates), 1),
        "sources": {"xgboost": wt["model_estimate_min"], "rag": wt["rag_median_min"]},
        "bottleneck_count": len(f["bottlenecks"]["flagged"]),
        "queue_trend": f["forecast"]["trend"],
    }}


# ── Validation & Critique Agent ──────────────────────────────────────────

def critique(state: SupervisorState) -> SupervisorState:
    fused, f = state["fused"], state["findings"]
    wt = f["waiting_time"]
    checks, score = [], 1.0

    if wt["rag_median_min"] is not None:
        rel_gap = fused["estimate_spread_min"] / max(fused["consensus_wait_min"], 1.0)
        agree = rel_gap <= 0.5
        checks.append({"check": "xgb_vs_rag_agreement", "passed": agree,
                       "detail": f"spread {fused['estimate_spread_min']}m "
                                 f"({rel_gap:.0%} of consensus)"})
        score -= 0.0 if agree else 0.3
    else:
        checks.append({"check": "rag_evidence", "passed": False,
                       "detail": "no RAG evidence for this category"})
        score -= 0.15

    hmm_state = next((s["state"] for s in f["bottlenecks"]["stations"]
                      if s["category"] == state["category"]), "unknown")
    consistent = not (hmm_state == "High" and f["forecast"]["trend"] == "falling"
                      and fused["consensus_wait_min"] < 15)
    checks.append({"check": "hmm_forecast_consistency", "passed": consistent,
                   "detail": f"HMM={hmm_state}, trend={f['forecast']['trend']}"})
    score -= 0.0 if consistent else 0.2

    if f["forecast"]["trend"] == "no-data":
        score -= 0.15

    return {"critique": {
        "checks": checks,
        "confidence": round(max(0.2, score), 2),
        "confidence_label": "high" if score >= 0.85 else "medium" if score >= 0.6 else "low",
    }}


# ── Decision Support Module ──────────────────────────────────────────────

def decision_support(state: SupervisorState) -> SupervisorState:
    f = state["findings"]
    decisions = []
    for b in f["bottlenecks"]["flagged"]:
        decisions.append({
            "priority": 1 if b["state"] == "High" else 2,
            "recommendation": f"Bottleneck at {b['category'].replace('_', ' ')}: "
                              f"{b['queue_depth']} waiting, HMM state {b['state']} "
                              f"(typical {b['state_mean_wait']}m). Escalate staffing.",
            "impact": "reduce station wait toward its Low-state mean",
        })
    for opt in f["resource_planning"]["options"]:
        decisions.append({
            "priority": 2 if opt["load_share_pct"] > 35 else 3,
            "recommendation": opt["option"],
            "impact": f"station holds {opt['load_share_pct']}% of clinic load "
                      f"({opt['queue_depth']} waiting)",
        })
    if f["forecast"]["trend"] == "rising":
        decisions.append({
            "priority": 1,
            "recommendation": f"Queue for {state['category'].replace('_', ' ')} is "
                              "forecast to rise within the next hour — pre-empt with "
                              "additional capacity now.",
            "impact": "avoid projected queue growth",
        })
    decisions.sort(key=lambda d: d["priority"])
    return {"decisions": decisions[:6]}


# ── Reporting & Visualization ────────────────────────────────────────────

def reporting(state: SupervisorState) -> SupervisorState:
    fused, crit = state["fused"], state["critique"]
    at = state["as_of"]
    summary = (
        f"As of {at:%A %H:%M}, expected wait for "
        f"{state['category'].replace('_', ' ')} at {state['clinic']} is "
        f"~{fused['consensus_wait_min']:.0f} min "
        f"(XGBoost {fused['sources']['xgboost']}m"
        + (f", historical RAG median {fused['sources']['rag']}m" if fused["sources"]["rag"] else "")
        + f"). Queue trend: {fused['queue_trend']}. "
        f"{fused['bottleneck_count']} station(s) flagged as bottlenecks. "
        f"Confidence: {crit['confidence_label']} ({crit['confidence']})."
    )

    if settings.anthropic_api_key:
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=settings.llm_model, max_tokens=400,
                                api_key=settings.anthropic_api_key)
            summary = llm.invoke(
                "Rewrite this clinic operations summary for a nurse manager. Keep every "
                f"number exactly as given, two short paragraphs max:\n\n{summary}"
            ).content
        except Exception:
            pass  # numeric summary stands on its own (NFR-4)

    alerts = [
        {"level": "critical" if d["priority"] == 1 else "warning",
         "message": d["recommendation"]}
        for d in state["decisions"] if d["priority"] <= 2
    ]
    return {"report": {
        "summary": summary,
        "as_of": at.isoformat(),
        "clinic": state["clinic"],
        "category": state["category"],
        "consensus_wait_min": fused["consensus_wait_min"],
        "confidence": crit,
        "findings": state["findings"],
        "recommendations": state["decisions"],
        "alerts": alerts,
    }}


def build_supervisor_graph():
    g = StateGraph(SupervisorState)
    g.add_node("supervisor", supervisor)
    g.add_node("waiting_time_agent", waiting_time_agent)
    g.add_node("bottleneck_agent", bottleneck_agent)
    g.add_node("resource_planning_agent", resource_planning_agent)
    g.add_node("predictive_agent", predictive_agent)
    g.add_node("result_fusion", result_fusion)
    g.add_node("critique", critique)
    g.add_node("decision_support", decision_support)
    g.add_node("reporting", reporting)

    g.set_entry_point("supervisor")
    for specialist in ("waiting_time_agent", "bottleneck_agent",
                       "resource_planning_agent", "predictive_agent"):
        g.add_edge("supervisor", specialist)          # parallel fan-out
        g.add_edge(specialist, "result_fusion")       # fan-in
    g.add_edge("result_fusion", "critique")
    g.add_edge("critique", "decision_support")
    g.add_edge("decision_support", "reporting")
    g.add_edge("reporting", END)
    return g.compile()


supervisor_graph = build_supervisor_graph()
