"""SOP & Healthcare Q&A agent — a genuine tool-calling LLM agent (LangGraph ReAct
pattern), unlike WaitEstimateGraph/supervisor_graph where the LLM only rephrases
numbers computed upstream. Here the LLM decides whether/what to search and
reasons over retrieved SOP passages; there is no template fallback, since
open-ended question answering can't be computed deterministically.

agent -(tool_calls?)-> tools -> agent -(no tool_calls)-> END
"""

from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from app.config import settings
from app.services.embedder import embed_one
from app.services.qdrant_service import get_client
from app.services.reranker import rerank_chunks
from app.services.sop_service import search_sop

SYSTEM_PROMPT = (
    "You are a clinical SOP assistant for an eye clinic. Answer only using passages "
    "returned by the search_sop_docs tool — call it at least once before answering, and "
    "call it again with a refined query if the first search doesn't cover the question. "
    "If the tool returns no relevant passages, say plainly that the knowledge base doesn't "
    "cover this question rather than guessing. Do not fabricate citations or sources — "
    "citations are attached automatically from what the tool actually returned."
)


class QAState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    citations: Annotated[list[dict], lambda a, b: a + b]


@tool
def search_sop_docs(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Search the SOP / healthcare-procedure knowledge base for passages relevant to `query`."""
    hits = search_sop(get_client(), embed_one(query), limit=settings.sop_retrieval_k)
    top = rerank_chunks(query, hits, top_n=settings.sop_rerank_top_n)

    if not top:
        content = "No relevant SOP passages found for this query."
        citations: list[dict] = []
    else:
        content = "\n\n---\n\n".join(
            f"[{i + 1}] {h['document_name']}\n{h['text']}" for i, h in enumerate(top)
        )
        citations = [
            {
                "title": h["document_name"],
                "snippet": h["text"][:280],
                "score": round(h["rerank_score"], 4),
            }
            for h in top
        ]

    return Command(update={
        "citations": citations,
        "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
    })


def call_model(state: QAState) -> QAState:
    llm = ChatOpenAI(model=settings.llm_model, max_tokens=500,
                      api_key=settings.openai_api_key).bind_tools([search_sop_docs])
    response = llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
    return {"messages": [response]}


def build_qa_graph():
    g = StateGraph(QAState)
    g.add_node("agent", call_model)
    g.add_node("tools", ToolNode([search_sop_docs]))
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


qa_graph = build_qa_graph()
