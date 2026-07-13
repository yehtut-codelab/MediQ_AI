"use client";

import { useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Source = { title: string; snippet: string; score?: number };
type Message = {
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  placeholder?: boolean;
};

const SUGGESTED = [
  "What is the SOP for pupil dilation before OCT imaging?",
  "When should a patient be escalated from triage to urgent consult?",
  "What are the fasting requirements before a fluorescein angiography?",
];

export default function SopQA() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  async function ask(question: string) {
    if (!question.trim() || loading) return;
    setMessages((m) => [...m, { role: "user", text: question }]);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch(`${API}/api/v1/qa/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      const data = await res.json();
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: data.answer,
          sources: data.sources,
          placeholder: data.status === "placeholder",
        },
      ]);
    } catch (err) {
      setMessages((m) => [...m, { role: "assistant", text: `Error: ${err}` }]);
    } finally {
      setLoading(false);
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    }
  }

  return (
    <>
      <div className="panel">
        <div className="panel-head">SOP &amp; Healthcare Q&amp;A</div>
        <div className="panel-body">
          <div className="qa-banner">
            Knowledge base integration pending — this page is wired to a placeholder
            endpoint (<code>POST /api/v1/qa/ask</code>). Connect the SOP RAG collection
            to make it live; the UI and API contract are ready.
          </div>

          <div className="qa-thread">
            {messages.length === 0 && (
              <div className="muted" style={{ padding: "16px 0" }}>
                Ask about clinic SOPs, protocols, or healthcare procedures. Try:
                <ul style={{ marginTop: 8, paddingLeft: 18 }}>
                  {SUGGESTED.map((q) => (
                    <li key={q} style={{ marginBottom: 6 }}>
                      <a
                        href="#"
                        style={{ color: "var(--mediq-blue)" }}
                        onClick={(e) => {
                          e.preventDefault();
                          ask(q);
                        }}
                      >
                        {q}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`qa-msg ${m.role}`}>
                <div className="qa-role">{m.role === "user" ? "You" : "SOP Assistant"}</div>
                <div>{m.text}</div>
                {m.placeholder && (
                  <div className="qa-chip">placeholder response — KB not connected</div>
                )}
                {m.sources && m.sources.length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    {m.sources.map((s, j) => (
                      <div key={j} className="qa-source">
                        <strong>{s.title}</strong> — {s.snippet}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {loading && <div className="muted">Searching knowledge base…</div>}
            <div ref={bottomRef} />
          </div>

          <form
            style={{ display: "flex", gap: 8, marginTop: 12 }}
            onSubmit={(e) => {
              e.preventDefault();
              ask(input);
            }}
          >
            <input
              style={{ flex: 1 }}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about SOPs or healthcare procedures…"
              maxLength={1000}
            />
            <button className="primary" style={{ width: 100 }} disabled={loading}>
              Ask
            </button>
          </form>
        </div>
      </div>
    </>
  );
}
