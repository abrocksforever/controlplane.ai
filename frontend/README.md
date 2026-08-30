# ControlPlane.ai — Demonstration Frontend

An interactive, responsive dark-mode Single-Page Application (SPA) and FastAPI gateway demonstrating the **ControlPlane.ai (PS1 Architecture)** Responsible AI Control Plane step-by-step.

---

## 🚀 Quick Launch

To start the web application and open the demonstration interface in your default browser:

```bash
python frontend/run.py
```

Or run via Uvicorn directly:

```bash
python frontend/server.py --port 8000
```

Open your browser at: **[http://localhost:8000](http://localhost:8000)**

---

## 🌟 Key Features

1. **Step-by-Step Demonstration Mode**:
   - Step through the 5-stage pipeline (**Stage 1: Pre-Guardrails** $\to$ **Stage 2: Primary LLM & CRAG** $\to$ **Stage 3A: Fast Heuristics & Statistical Scorer** $\to$ **Stage 3B: RAG Grounding Verification** $\to$ **Stage 3C: AI-as-a-Judge** $\to$ **Stage 4: Policy Arbitration** $\to$ **Stage 5: Delivered Output & Cryptographic Governance**).
   - Interactive Stepper bar with stage spotlights, explanations, and metrics.
   - Auto Play mode and step walkthrough controls.

2. **One-Click Demonstration Presets**:
   - **Scenario 1**: Standard Safe Policy Inquiry (India UPI) $\to$ `ALLOW`
   - **Scenario 2**: Financial Transaction Trigger ($3,500 Payout) $\to$ `HITL` (Forced Escalation)
   - **Scenario 3**: Guest PII In-Flight Sanitization (Credit Card & Email Masking) $\to$ Redaction & Safe processing
   - **Scenario 4**: Ungrounded Policy Contradiction (45-Day cash refund trap) $\to$ `HITL` / `BLOCK`
   - **Scenario 5**: Adversarial Prompt Injection (DAN / Developer mode) $\to$ `BLOCK` (Early Termination)
   - **Scenario 6**: Fabricated Absolute Guarantee $\to$ RAG Mismatch

3. **Human-in-the-Loop (HITL) Review Queue**:
   - Real-time review queue table with pending quarantined tickets.
   - Reviewer actions: `ALLOW`, `EDIT`, `BLOCK`.
   - Active learning feedback calibration (Approval rate, Edit rate, Block rate, Sensitivity tuning recommendations).

4. **Authoritative Knowledge Base Explorer**:
   - Browse and search all 20 authoritative Airbnb policy documents.
   - Filter by category, product, and audience.

5. **Cryptographic SHA-256 Audit Log Verifier**:
   - Live immutable hash chain explorer ($H_i = \text{SHA256}(H_{i-1} + \text{Payload})$).
   - One-click cryptographic proof verification confirming zero tampering.

6. **System Architecture & Mathematical Reference**:
   - Formulas, SLA latency targets, score weights, and 3-tier routing matrix thresholds.

---

## 📁 Directory Structure

```
frontend/
├── server.py             # FastAPI backend API gateway & static file server
├── run.py                # One-click browser launch script
├── test_frontend.py      # Automated endpoint test suite
├── README.md             # Frontend documentation
└── static/
    ├── index.html        # Single Page Application HTML
    ├── styles.css        # Custom dark-theme styling, glassmorphism, animations
    └── app.js            # Stepper logic, state management, API interactions
```

---

## 🧪 Testing

Run the automated endpoint test suite:

```bash
python frontend/test_frontend.py
```
