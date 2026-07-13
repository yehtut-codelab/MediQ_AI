"""Rule-based ophthalmic triage over free-text presentation.

Deterministic and auditable by design: every priority decision cites the
red-flag phrases that triggered it. No LLM in the decision path — triage is
safety-critical and must be explainable to clinical staff.

Priorities follow standard ophthalmic triage bands:
  P1 immediate — sight-threatening, seen ahead of all queues
  P2 urgent    — same-session escalation, seen ahead of routine
  P3 routine   — normal queue order
"""

import re
from typing import NamedTuple


class TriageResult(NamedTuple):
    priority: int  # 1 | 2 | 3
    label: str
    reasons: list[str]


PRIORITY_LABELS = {1: "P1 — Immediate", 2: "P2 — Urgent", 3: "P3 — Routine"}

# (compiled pattern, priority, human-readable red flag)
_RULES: list[tuple[re.Pattern, int, str]] = [
    # ---- P1: sight-threatening presentations ----
    (re.compile(r"sudden.{0,30}(vision loss|loss of vision|blur)|"
                r"(vision|sight).{0,20}(loss|gone|black).{0,20}sudden", re.I),
     1, "sudden vision loss"),
    (re.compile(r"chemical|acid|alkali|splash", re.I), 1, "chemical injury"),
    (re.compile(r"trauma|injur|hit in|blow to|foreign body|penetrat", re.I),
     1, "ocular trauma / foreign body"),
    (re.compile(r"(flash\w*).{0,60}(floater\w*)|(floater\w*).{0,60}(flash\w*)|"
                r"curtain|shadow.{0,20}(vision|field)", re.I),
     1, "flashes + floaters / curtain (retinal detachment risk)"),
    (re.compile(r"severe.{0,20}(eye )?pain|excruciating|"
                r"(pain|ache).{0,40}(nausea|vomit|halos)", re.I),
     1, "acute severe pain (angle-closure risk)"),
    (re.compile(r"(post.?op|surgery|operation).{0,60}"
                r"(pain|worse|drop|loss|discharge|pus)", re.I),
     1, "post-operative deterioration (endophthalmitis risk)"),
    # ---- P2: urgent, same-session escalation ----
    (re.compile(r"new.{0,20}floater|floater.{0,20}(new|recent|since)", re.I),
     2, "new-onset floaters"),
    (re.compile(r"(double vision|diplopia)", re.I), 2, "new diplopia"),
    (re.compile(r"red eye|redness|conjunctivitis|discharge", re.I),
     2, "acute red eye / discharge"),
    (re.compile(r"(eye )?pressure|high iop|raised iop", re.I),
     2, "raised pressure symptoms"),
    (re.compile(r"halos?\b", re.I), 2, "halos around lights"),
    (re.compile(r"(vision|sight).{0,30}(worse|deteriorat|drop)", re.I),
     2, "recent visual deterioration"),
]


def triage(*texts: str | None) -> TriageResult:
    """Scan diagnosis / history / current-issue text; best (lowest) priority wins."""
    blob = " ".join(t for t in texts if t)
    priority, reasons = 3, []
    for pattern, p, reason in _RULES:
        if pattern.search(blob):
            reasons.append(reason)
            priority = min(priority, p)
    if priority == 3:
        reasons = ["no red flags identified"]
    return TriageResult(priority, PRIORITY_LABELS[priority], reasons)
