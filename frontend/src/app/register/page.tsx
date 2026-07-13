"use client";

import { useEffect, useState } from "react";
import SuggestField from "@/components/SuggestField";
import { BAND_LABEL, bandFor } from "@/lib/band";
import { CURRENT_ISSUE_OPTIONS, DIAGNOSIS_OPTIONS, HISTORY_OPTIONS } from "@/lib/clinicalOptions";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const CLINICS = ["TTSH Eye Centre", "Clinic 1A"];
const SERVICE_TYPES = [
  "Consultation", "OCT", "VA", "HVF", "Orthoptic", "Biometry", "PAT",
  "Diagnostic", "Financial Con", "Pre Consultation Test",
];

type PathwayStep = {
  station: string;
  eta_offset_min: number;
  expected_wait_min: number;
  expected_service_min: number;
  note: string | null;
};

type ForecastBucket = { minutes_ahead: number; queue_depth: number };

type RegisteredPatient = {
  patient_id: number;
  display_name: string | null;
  clinic: string;
  service_type: string;
  arrival_datetime: string;
  appointment_datetime: string | null;
  queue_position: number;
  priority: number;
  priority_label: string;
  triage_reasons: string[];
  pathway_label: string | null;
  pathway: PathwayStep[];
  total_visit_min: number | null;
  estimated_wait_min: number;
  range_min: [number, number];
  status_band: "green" | "amber" | "red";
  explanation: string;
  model_category: string | null;
  model_estimate_min: number | null;
  model_mae_min: number | null;
  model_queue_depth: number | null;
  model_congestion_state: "Low" | "Medium" | "High" | "unknown" | null;
  model_congestion_mean_wait_min: number | null;
  model_forecast: ForecastBucket[];
  model_forecast_trend: "rising" | "falling" | "stable" | "no-data" | null;
  model_replayed_at: string | null;
  model_is_analog_replay: boolean;
  model_replay_note: string | null;
  model_unavailable_reason: string | null;
};

const PRIORITY_COLOR: Record<number, string> = {
  1: "var(--mediq-red, #c0392b)",
  2: "#b9770e",
  3: "#1e8449",
};

const CONGESTION_COLOR: Record<string, string> = {
  High: "var(--mediq-red)",
  Medium: "var(--mediq-gold)",
  Low: "var(--mediq-green)",
  unknown: "#999",
};

const TREND_ARROW: Record<string, string> = {
  rising: "↑ rising",
  falling: "↓ falling",
  stable: "→ stable",
  "no-data": "— no data",
};

function PriorityBadge({ p, label }: { p: number; label: string }) {
  return (
    <span
      style={{
        background: PRIORITY_COLOR[p] ?? PRIORITY_COLOR[3],
        color: "#fff",
        borderRadius: 4,
        padding: "2px 8px",
        fontSize: 12,
        fontWeight: 700,
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </span>
  );
}

function nowLocal(): string {
  const d = new Date();
  d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
  return d.toISOString().slice(0, 16);
}

export default function RegisterPatient() {
  const [name, setName] = useState("");
  const [patientId, setPatientId] = useState("");
  const [clinic, setClinic] = useState(CLINICS[0]);
  const [serviceType, setServiceType] = useState(SERVICE_TYPES[0]);
  const [arrival, setArrival] = useState(nowLocal);
  const [appointment, setAppointment] = useState("");
  const [diagnosis, setDiagnosis] = useState("");
  const [history, setHistory] = useState("");
  const [currentIssue, setCurrentIssue] = useState("");
  const [registered, setRegistered] = useState<RegisteredPatient | null>(null);
  const [queue, setQueue] = useState<RegisteredPatient[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function refreshQueue() {
    try {
      const res = await fetch(`${API}/api/v1/patients`);
      if (res.ok) setQueue(await res.json());
    } catch {
      /* queue list is non-critical */
    }
  }

  useEffect(() => {
    refreshQueue();
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API}/api/v1/patients/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientId ? Number(patientId) : null,
          display_name: name || null,
          clinic,
          service_type: serviceType,
          arrival_datetime: `${arrival}:00+08:00`,
          appointment_datetime: appointment ? `${appointment}:00+08:00` : null,
          diagnosis: diagnosis || null,
          medical_history: history || null,
          current_issue: currentIssue || null,
        }),
      });
      if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
      setRegistered(await res.json());
      setName("");
      setPatientId("");
      setAppointment("");
      setArrival(nowLocal());
      setDiagnosis("");
      setHistory("");
      setCurrentIssue("");
      refreshQueue();
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <div className="panel">
        <div className="panel-head">New Patient Entry — Registration</div>
        <div className="panel-body">
          <form className="form-grid" onSubmit={submit}>
            <div>
              <label>Patient Name (demo only, not persisted)</label>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Tan Ah Kow"
                maxLength={80}
              />
            </div>
            <div>
              <label>Patient ID (blank = auto-generate)</label>
              <input
                value={patientId}
                onChange={(e) => setPatientId(e.target.value.replace(/\D/g, ""))}
                placeholder="9-digit ID"
                inputMode="numeric"
              />
            </div>
            <div>
              <label>Clinic</label>
              <select value={clinic} onChange={(e) => setClinic(e.target.value)}>
                {CLINICS.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </div>
            <div>
              <label>Service Type</label>
              <select value={serviceType} onChange={(e) => setServiceType(e.target.value)}>
                {SERVICE_TYPES.map((s) => (
                  <option key={s}>{s}</option>
                ))}
              </select>
            </div>
            <div>
              <label>Arrival (SGT)</label>
              <input
                type="datetime-local"
                value={arrival}
                onChange={(e) => setArrival(e.target.value)}
                required
              />
            </div>
            <div>
              <label>Appointment (optional)</label>
              <input
                type="datetime-local"
                value={appointment}
                onChange={(e) => setAppointment(e.target.value)}
              />
            </div>
            <SuggestField
              id="diagnosis"
              label="Diagnosis"
              value={diagnosis}
              onChange={setDiagnosis}
              options={DIAGNOSIS_OPTIONS}
              placeholder="Pick a common diagnosis or type your own…"
              maxLength={500}
            />
            <SuggestField
              id="medical-history"
              label="Medical History"
              value={history}
              onChange={setHistory}
              options={HISTORY_OPTIONS}
              placeholder="Pick from the list or type your own…"
              maxLength={1000}
            />
            <SuggestField
              id="current-issue"
              label="Current Issue"
              value={currentIssue}
              onChange={setCurrentIssue}
              options={CURRENT_ISSUE_OPTIONS}
              placeholder="Pick a common presentation or type your own…"
              maxLength={1000}
            />
            <button className="primary" disabled={loading}>
              {loading ? "Registering patient & computing estimates…" : "Register Patient & Estimate Wait"}
            </button>
          </form>
          {error && <p style={{ color: "var(--mediq-red)", marginTop: 10 }}>{error}</p>}
        </div>
      </div>

      {registered && (
        <div className="panel">
          <div className="panel-head">Registration Ticket</div>
          <div className="panel-body" aria-live="polite">
            <div className={`estimate-card ${registered.status_band}`}>
              <div style={{ display: "flex", gap: 32, alignItems: "baseline" }}>
                <div>
                  <div className="muted">Queue position</div>
                  <div className="estimate-value">Q{registered.queue_position}</div>
                </div>
                <div>
                  <div className="muted">Estimated wait</div>
                  <div className={`estimate-value band-${registered.status_band}`}>
                    {Math.round(registered.estimated_wait_min)}
                    <span style={{ fontSize: 18 }}> min</span>
                    <span className={`band-chip ${registered.status_band}`}>{BAND_LABEL[registered.status_band]}</span>
                  </div>
                </div>
              </div>
              <div className="muted" style={{ marginTop: 6 }}>
                {registered.display_name ? `${registered.display_name} · ` : ""}
                Patient #{registered.patient_id} · {registered.service_type} ·{" "}
                {registered.clinic} · range {registered.range_min[0]}–{registered.range_min[1]} min
              </div>
              <div style={{ marginTop: 10, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                <PriorityBadge p={registered.priority} label={registered.priority_label} />
                <span className="muted" style={{ fontSize: 13 }}>
                  {registered.triage_reasons.join(" · ")}
                </span>
              </div>
              <p style={{ marginTop: 8 }}>{registered.explanation}</p>
            </div>
          </div>
        </div>
      )}

      {registered && (
        <div className="panel">
          <div className="panel-head">Model-Based Queue State — XGBoost · HMM · LSTM</div>
          <div className="panel-body" aria-live="polite">
            {registered.model_unavailable_reason && (
              <p className="muted">{registered.model_unavailable_reason}</p>
            )}
            {registered.model_estimate_min != null && (
              <>
                {registered.model_is_analog_replay && (
                  <p className="qa-banner" style={{ marginBottom: 12 }}>
                    {registered.model_replay_note}
                  </p>
                )}
                <div style={{ display: "flex", gap: 40, marginBottom: 12, flexWrap: "wrap" }}>
                  <div>
                    <div className="muted">Model wait estimate ({registered.model_category?.replace(/_/g, " ")})</div>
                    <div className={`estimate-value band-${bandFor(registered.model_estimate_min)}`}>
                      {Math.round(registered.model_estimate_min)}
                      <span style={{ fontSize: 18 }}> min</span>
                      <span className={`band-chip ${bandFor(registered.model_estimate_min)}`}>
                        {BAND_LABEL[bandFor(registered.model_estimate_min)]}
                      </span>
                    </div>
                    <div className="muted" style={{ fontSize: 11 }}>
                      XGBoost · val MAE {registered.model_mae_min} min
                    </div>
                  </div>
                  <div>
                    <div className="muted">Queue depth now</div>
                    <div className="kpi-secondary">{registered.model_queue_depth} waiting</div>
                  </div>
                  <div>
                    <div className="muted">Congestion state (HMM)</div>
                    <div
                      className="kpi-secondary"
                      style={{ color: CONGESTION_COLOR[registered.model_congestion_state ?? "unknown"] }}
                    >
                      {registered.model_congestion_state}
                    </div>
                    <div className="muted" style={{ fontSize: 11 }}>
                      {registered.model_congestion_mean_wait_min != null
                        ? `state mean ${registered.model_congestion_mean_wait_min}m`
                        : "insufficient recent data"}
                    </div>
                  </div>
                  <div>
                    <div className="muted">Forecast trend (LSTM)</div>
                    <div className="kpi-secondary">
                      {TREND_ARROW[registered.model_forecast_trend ?? "no-data"]}
                    </div>
                  </div>
                </div>

                {registered.model_forecast.length > 0 && (
                  <>
                    <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
                      Queue depth forecast — next{" "}
                      {registered.model_forecast[registered.model_forecast.length - 1].minutes_ahead} min
                    </div>
                    <div className="hourly-bars" style={{ maxWidth: 320 }}>
                      {registered.model_forecast.map((f, i) => {
                        const max = Math.max(...registered.model_forecast.map((x) => x.queue_depth), 1);
                        return (
                          <div
                            key={i}
                            className="hourly-bar"
                            role="img"
                            aria-label={`+${f.minutes_ahead} minutes: ${f.queue_depth} patients in queue`}
                            title={`+${f.minutes_ahead} min: ${f.queue_depth}`}
                            style={{
                              height: `${Math.max(4, (f.queue_depth / max) * 40)}px`,
                              background: "var(--mediq-blue)",
                            }}
                          />
                        );
                      })}
                    </div>
                  </>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {registered && registered.pathway.length > 0 && (
        <div className="panel">
          <div className="panel-head">
            Predicted Visit Pathway — {registered.pathway_label}
            {registered.total_visit_min != null &&
              ` · ~${Math.round(registered.total_visit_min)} min total`}
          </div>
          <div className="panel-body">
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "stretch" }}>
              {registered.pathway.map((step, i) => (
                <div
                  key={i}
                  style={{
                    border: "1px solid var(--border, #ddd)",
                    borderRadius: 6,
                    padding: "8px 12px",
                    minWidth: 150,
                  }}
                >
                  <div style={{ fontWeight: 700 }}>
                    {i + 1}. {step.station}
                  </div>
                  <div className="muted" style={{ fontSize: 13 }}>
                    starts ~T+{Math.round(step.eta_offset_min)} min
                    <br />
                    wait ~{Math.round(step.expected_wait_min)} min · service ~
                    {Math.round(step.expected_service_min)} min
                    {step.note && (
                      <>
                        <br />
                        {step.note}
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
            <p className="muted" style={{ marginTop: 10, fontSize: 13 }}>
              Pathway and durations from observed patient journeys (median per station).
            </p>
          </div>
        </div>
      )}

      {queue.length > 0 && (
        <div className="panel">
          <div className="panel-head">
            Registered Today ({queue.length}) — triage order
          </div>
          <div className="panel-body">
            <table className="evidence-table">
              <thead>
                <tr>
                  <th>Priority</th>
                  <th>Q#</th>
                  <th>Patient</th>
                  <th>Clinic</th>
                  <th>Service</th>
                  <th>Pathway</th>
                  <th>Arrival</th>
                  <th>Est. wait</th>
                </tr>
              </thead>
              <tbody>
                {[...queue]
                  .sort(
                    (a, b) =>
                      a.priority - b.priority ||
                      a.arrival_datetime.localeCompare(b.arrival_datetime)
                  )
                  .map((p, i) => (
                    <tr key={i}>
                      <td>
                        <PriorityBadge p={p.priority} label={`P${p.priority}`} />
                      </td>
                      <td>Q{p.queue_position}</td>
                      <td>{p.display_name || `#${p.patient_id}`}</td>
                      <td>{p.clinic}</td>
                      <td>{p.service_type}</td>
                      <td className="muted" style={{ fontSize: 13 }}>
                        {p.pathway.map((s) => s.station).join(" → ") || "—"}
                      </td>
                      <td>{p.arrival_datetime.slice(11, 16)}</td>
                      <td className={`band-${p.status_band}`} style={{ fontWeight: 600 }}>
                        {Math.round(p.estimated_wait_min)} min
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}
