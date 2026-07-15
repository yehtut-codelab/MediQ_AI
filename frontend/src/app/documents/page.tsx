"use client";

import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type DocStatus = "uploaded" | "processing" | "ingested" | "failed";

type DocumentOut = {
  id: string;
  name: string;
  ext: string;
  size_bytes: number;
  status: DocStatus;
  chunk_count: number;
  error_message: string | null;
  uploaded_at: string;
  ingested_at: string | null;
};

const STATUS_BAND: Record<DocStatus, "green" | "amber" | "red"> = {
  ingested: "green",
  processing: "amber",
  uploaded: "amber",
  failed: "red",
};

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentOut[] | null>(null);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function refresh() {
    try {
      const res = await fetch(`${API}/api/v1/documents`);
      if (!res.ok) throw new Error(`API ${res.status}`);
      setDocuments(await res.json());
      setError("");
    } catch (err) {
      setError(String(err));
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleUpload() {
    const file = fileRef.current?.files?.[0];
    if (!file || uploading) return;

    setUploading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/api/v1/documents/upload`, { method: "POST", body: form });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail ?? `API ${res.status}`);
      }
      if (fileRef.current) fileRef.current.value = "";
      await refresh();
    } catch (err) {
      setError(String(err));
    } finally {
      setUploading(false);
    }
  }

  async function handleReingest(id: string) {
    setBusyId(id);
    setError("");
    try {
      const res = await fetch(`${API}/api/v1/documents/${id}/reingest`, { method: "POST" });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail ?? `API ${res.status}`);
      }
      await refresh();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusyId(null);
    }
  }

  async function handleDelete(id: string, name: string) {
    if (!confirm(`Remove "${name}" from the knowledge base? This deletes its chunks from Qdrant too.`)) return;
    setBusyId(id);
    setError("");
    try {
      const res = await fetch(`${API}/api/v1/documents/${id}`, { method: "DELETE" });
      if (!res.ok && res.status !== 204) throw new Error(`API ${res.status}`);
      await refresh();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="panel">
      <div className="panel-head">Knowledge Base — Document Management</div>
      <div className="panel-body">
        <div className="doc-upload">
          <input ref={fileRef} type="file" accept=".pdf,.docx,.txt" disabled={uploading} />
          <button className="primary" style={{ width: 160 }} onClick={handleUpload} disabled={uploading}>
            {uploading ? "Uploading…" : "Upload & Ingest"}
          </button>
          <div className="doc-upload-hint">
            PDF, DOCX, or TXT. The file is chunked, embedded, and upserted into the{" "}
            <code>mediq_sop_docs</code> Qdrant collection immediately — used by the{" "}
            <a href="/qa" style={{ color: "var(--mediq-blue)" }}>SOP Q&amp;A</a> agent.
          </div>
        </div>

        {error && <p style={{ color: "var(--mediq-red)", marginBottom: 12 }}>{error}</p>}

        {!documents && !error && <p className="muted">Loading documents…</p>}

        {documents && documents.length === 0 && (
          <p className="muted">No documents yet — upload one above to start building the knowledge base.</p>
        )}

        {documents && documents.length > 0 && (
          <table className="evidence-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Size</th>
                <th>Status</th>
                <th>Chunks</th>
                <th>Uploaded</th>
                <th>Ingested</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((d) => (
                <tr key={d.id}>
                  <td>
                    <div className="doc-name">{d.name}</div>
                    {d.status === "failed" && d.error_message && (
                      <div className="doc-error">{d.error_message}</div>
                    )}
                  </td>
                  <td>{d.ext.replace(".", "").toUpperCase()}</td>
                  <td>{formatSize(d.size_bytes)}</td>
                  <td>
                    <span className={`band-chip ${STATUS_BAND[d.status]}`}>{d.status}</span>
                  </td>
                  <td>{d.chunk_count || "—"}</td>
                  <td>{formatTime(d.uploaded_at)}</td>
                  <td>{formatTime(d.ingested_at)}</td>
                  <td>
                    <div className="doc-actions">
                      {d.status === "failed" && (
                        <button
                          className="btn-small"
                          disabled={busyId === d.id}
                          onClick={() => handleReingest(d.id)}
                        >
                          {busyId === d.id ? "Retrying…" : "Retry"}
                        </button>
                      )}
                      <button
                        className="btn-small danger"
                        disabled={busyId === d.id}
                        onClick={() => handleDelete(d.id, d.name)}
                      >
                        {busyId === d.id ? "…" : "Delete"}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
