"use client";

import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Source = { title: string; snippet: string; score?: number };
type Message = {
  id: number;
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  placeholder?: boolean;
  isError?: boolean;
  time: string;
};

const SUGGESTED = [
  "What is the SOP for pupil dilation before OCT imaging?",
  "When should a patient be escalated from triage to urgent consult?",
  "What are the fasting requirements before a fluorescein angiography?",
];

function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

let nextId = 1;

export default function SopQA() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  function autoResize() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  }

  async function ask(question: string) {
    if (!question.trim() || loading) return;
    setMessages((m) => [...m, { id: nextId++, role: "user", text: question, time: now() }]);
    setInput("");
    setLoading(true);
    requestAnimationFrame(autoResize);

    try {
      const res = await fetch(`${API}/api/v1/qa/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (res.status === 503) {
        setMessages((m) => [
          ...m,
          {
            id: nextId++,
            role: "assistant",
            text: "SOP Q&A is not available — the backend needs OPENAI_API_KEY configured to answer questions.",
            isError: true,
            time: now(),
          },
        ]);
        return;
      }
      if (!res.ok) throw new Error(`API ${res.status}`);

      const data = await res.json();
      setMessages((m) => [
        ...m,
        {
          id: nextId++,
          role: "assistant",
          text: data.answer,
          sources: data.sources,
          placeholder: data.status === "placeholder",
          time: now(),
        },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          id: nextId++,
          role: "assistant",
          text: `Something went wrong reaching the SOP knowledge base: ${err}`,
          isError: true,
          time: now(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask(input);
    }
  }

  return (
    <div className="panel qa-page">
      <div className="panel-head">SOP &amp; Healthcare Q&amp;A</div>
      <div className="panel-body">
        <div className="qa-banner">
          Answers are grounded in the <code>mediq_sop_docs</code> Qdrant collection —
          run <code>scripts/ingest_sop_docs.py</code> to add SOP documents. Requires
          <code> OPENAI_API_KEY</code> to be configured on the backend.
        </div>

        <div className="qa-toolbar">
          <button
            className="qa-clear-btn"
            onClick={() => setMessages([])}
            disabled={messages.length === 0 || loading}
          >
            New chat
          </button>
        </div>

        <div className="qa-thread">
          {messages.length === 0 && (
            <div className="qa-empty">
              <h3>Ask about clinic SOPs, protocols, or healthcare procedures</h3>
              <div className="muted">Try one of these, or type your own question below</div>
              <div className="qa-suggestions">
                {SUGGESTED.map((q) => (
                  <button key={q} className="qa-suggestion-btn" onClick={() => ask(q)}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m) => (
            <div key={m.id} className={`qa-msg ${m.role}${m.isError ? " error" : ""}`}>
              <div className="qa-msg-meta">
                <span className="qa-role">{m.role === "user" ? "You" : "SOP Assistant"}</span>
                <span className="qa-time">{m.time}</span>
              </div>
              <div className="qa-text">{m.text}</div>
              {m.placeholder && (
                <div className="qa-chip">placeholder response — KB not connected</div>
              )}
              {m.sources && m.sources.length > 0 && (
                <details className="qa-sources">
                  <summary>{m.sources.length} source{m.sources.length > 1 ? "s" : ""}</summary>
                  <div className="qa-sources-list">
                    {m.sources.map((s, j) => (
                      <div key={j} className="qa-source">
                        <strong>{s.title}</strong> — {s.snippet}
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          ))}

          {loading && (
            <div className="qa-msg assistant">
              <div className="qa-msg-meta">
                <span className="qa-role">SOP Assistant</span>
              </div>
              <div className="qa-typing">
                <span></span><span></span><span></span>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <form
          className="qa-input-row"
          onSubmit={(e) => {
            e.preventDefault();
            ask(input);
          }}
        >
          <textarea
            ref={textareaRef}
            className="qa-textarea"
            rows={1}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              autoResize();
            }}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about SOPs or healthcare procedures… (Enter to send, Shift+Enter for a new line)"
            maxLength={1000}
          />
          <button className="primary qa-send-btn" disabled={loading || !input.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
