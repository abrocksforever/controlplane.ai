# ControlPlane.ai — Dashboard Frontend

A live, judge-facing dashboard for the PS1 Responsible AI Control Plane. It does not
reimplement or mock any scoring logic — `app.py` is a thin FastAPI shell around the
existing `pipeline.py`, so every number on screen comes from a real run of Stage 1
through Stage 5.

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Open **http://localhost:8000**. No API key is required — if `GROQ_API_KEY` isn't set,
the primary LLM and AI-judge stages fall back to their built-in defaults exactly as
`pipeline.py` already does, so the dashboard works fully offline. Set the key first if
you want live Groq generations and judge reasoning:

```bash
export GROQ_API_KEY="gsk_your_key_here"   # optional
python app.py
```

## What's in the dashboard

**Live Inspection** — type a prompt (or click one of the six scenarios lifted straight
from `demo.py`) and watch it travel across the seven real pipeline checkpoints:
Guardrails → Primary LLM → Fast Checks → RAG Grounding → AI Judge → Arbitration →
Governance. Click any node to see that stage's raw telemetry. The composite-risk gauge's
colored zones are drawn at the exact `ALLOW`/`HITL`/`BLOCK` thresholds from
`models.py::Config` (2.5 and 7.0), and the risk-contribution bars use the exact stage
weights (0.25 / 0.15 / 0.35 / 0.25).

**Audit Ledger** — renders `audit_log.jsonl` as a hash-chained timeline and calls
`verify_audit_log_integrity()` on demand so you can prove tamper-evidence live, in front
of judges.

**HITL Queue** — every ticket `arbitrator.py` quarantines shows up here for a reviewer to
Approve, Edit & release, or Override & block, wired to `HITLQueueManager.resolve_ticket`.

A **Reset Demo** button in the top bar clears the audit ledger and ticket queue so you
can start a clean run mid-presentation.

## API surface (new — added for the frontend)

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | Runs `run_controlplane()` for a prompt, returns the full `PipelineOutput` |
| `GET /api/scenarios` | The 6 benchmark scenarios from `demo.py` |
| `GET /api/audit-log` | Parsed `audit_log.jsonl` + live chain-integrity check |
| `POST /api/audit-log/verify` | Re-runs `verify_audit_log_integrity()` |
| `GET /api/hitl-queue` | Pending + resolved tickets, plus tuning metrics |
| `POST /api/hitl-resolve` | Approve / Edit / Override a ticket |
| `POST /api/reset-demo` | Clears the ledger and queue for a fresh run |
| `GET /api/health` | Chain status + pending ticket count, polled every 15s |

`demo.py` and the `pytest` suite are untouched and still work standalone — the frontend
is purely additive.
