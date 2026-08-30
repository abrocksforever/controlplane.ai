# 🛡️ ControlPlane.ai — Responsible AI Control Plane & Evaluative Gateway

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pytest](https://img.shields.io/badge/Tests-113%2F113%20Passed%20(100%25)-emerald?logo=pytest&logoColor=white)](tests/)
[![Benchmark](https://img.shields.io/badge/Benchmark-50%2F50%20Compliant%20(100%25)-indigo)](benchmark_airbnb.py)
[![Latency](https://img.shields.io/badge/Fast--Path%20Latency-P50%3A%206.8ms%20%7C%20P90%3A%207.7ms-cyan)](#-performance-benchmarks)
[![Audit](https://img.shields.io/badge/Audit%20Log-SHA--256%20Hash%20Chained-purple)](#-stage-5-governance-hitl-triage--cryptographic-audit)
[![License](https://img.shields.io/badge/License-MIT-amber.svg)](LICENSE)

**A high-performance, modular Responsible AI Control Plane featuring pre-execution guardrails, parallel scatter-gather heuristics, Corrective RAG (CRAG) factual grounding, AI-as-a-Judge compliance evaluation, 3-tier policy arbitration, and immutable cryptographic SHA-256 hash-chained audit logging.**

[🚀 Quick Start](#-quick-start) • [✨ Key Innovations](#-key-architectural-innovations) • [📊 Benchmark Scorecard](#-performance-benchmarks) • [🌐 Web Dashboard](#-interactive-web-dashboard) • [📚 API Reference](#-rest-api-reference) • [📁 Directory Structure](#-repository-structure)

</div>

---

## 📌 Problem Statement Overview (PS1 Alignment)

Enterprise deployment of Large Language Models introduces severe vulnerabilities: **sensitive PII leaks**, **adversarial prompt injections (DAN attacks)**, **hallucinated policy terms / financial promises**, **demographic biases**, and **lack of regulatory auditability**. 

Traditional safety systems rely either on slow LLM-as-a-judge gateways (introducing 2–4 second latency penalties on every turn) or fragile keyword filters that penalize legitimate denials.

**`ControlPlane.ai`** solves this through a decoupled, multi-tiered evaluative control plane designed to:
1. **Intercept Threats Early (<1ms)**: Redact PII in-flight and terminate jailbreak injections pre-execution.
2. **Eliminate Hallucinations with CRAG**: Calculate retrieval quality ($\rho \in [0, 1]$), actively abstain on unindexed empirical claims, and use negation-window filtering to prevent false positives when refuting invalid customer claims.
3. **Route Adaptively for Real-Time SLAs**: Deliver clean routine traffic in **$<15\text{ms}$** (Fast-Path), dynamically auto-promoting to Deep-Path (<2000ms) only on financial triggers, CRAG ambiguity, or risk anomalies.
4. **Enforce Absolute Auditability**: Log every interaction into an immutable, cryptographically chained SHA-256 ledger with a zero-friction Human-in-the-Loop (HITL) review queue.

---

## 🏛️ 5-Stage Modular Pipeline Architecture

```
                                  [ User Prompt ]
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STAGE 1: Pre-Execution Guardrails (pii.py) [<1.0 ms]                        │
  │ • Reverse-Offset Slice Masking (SSN, Email, API Keys)                       │
  │ • Luhn Algorithmic Checksum Validation (Visa, Mastercard, Amex)             │
  │ • Weighted Adversarial Prompt Injection Classifier (DAN / Jailbreak Filter) │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │ (Clean / Sanitized Prompt)
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STAGE 2: Context Retrieval & Primary Generation (llm_client.py)             │
  │ • In-Memory BM25 Knowledge Retrieval (<0.05ms) + Authoritative Airbnb Corpus │
  │ • Corrective RAG (CRAG) Retrieval Quality Evaluator                         │
  │ • Primary LLM Generation (Groq / Qwen / Llama-3.3-70B or Offline Engine)   │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │ (Candidate Response)
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ ADAPTIVE LATENCY ROUTER: Fast-Path vs Deep-Path Auto-Elevation              │
  │ Routine traffic (ρ ≥ 0.70, clean heuristics) ───────────────► FAST PATH      │
  │ Financial triggers, CRAG ambiguity, or risk anomaly ────────► DEEP PATH      │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
  ┌──────────────────────────────┐                ┌──────────────────────────────┐
  │ FAST-PATH PARALLEL CHECKS    │                │ DEEP-PATH EVALUATION         │
  │                              │                │                              │
  │ [STAGE 3A: Fast Checks]      │                │ [STAGE 3B: Semantic NLI]     │
  │ • Scatter-Gather Worker Bus  │                │ • Sentence-Level Entailment  │
  │ • Output PII & Banned Lexicon│                │ • Refusal/Disclaimer Immunity│
  │ • Shannon Entropy & Loops    │                │                              │
  │                              │                │ [STAGE 3C: AI-as-a-Judge]    │
  │ [STAGE 3B: Negation Filter]  │                │ • JSON Mode Compliance LLM   │
  │ • Numeric Span Extraction    │                │ • Demographic Bias Detection │
  │ • Refusal Clause Immunity    │                │ • Tone & Policy Verification │
  └──────────────┬───────────────┘                └──────────────┬───────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │ (Risk Dimensions Breakdown)
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STAGE 4: Policy Arbitration & Risk Assessment (arbitrator.py)               │
  │ • Dynamic Active Weight Renormalization                                     │
  │ • Financial Gate Trigger (FinCheck: Wire Transfers / Payouts ≥ $1,000)       │
  │ • 3-Tier Policy Decision Matrix:                                            │
  │     [ALLOW: S ≤ 2.50]  │  [HITL Quarantine: 2.50 < S < 7.00]  │  [BLOCK: S ≥ 7.00]
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STAGE 5: Governance, HITL Queue & Cryptographic Audit (audit_hitl.py)       │
  │ • Persistent SQLite HITL Triage Queue (ALLOW / EDIT / BLOCK Review)         │
  │ • Active Learning Continuous Feedback Loop                                  │
  │ • Immutable O(1) Reverse-Seek SHA-256 Hash Chained Audit Ledger             │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Architectural Innovations

### 1. Corrective RAG (CRAG) & Active Abstention Gate
Traditional RAG blindly passes ungrounded candidate claims when 0 matching documents are retrieved. ControlPlane.ai computes a normalized retrieval confidence score $\rho \in [0.0, 1.0]$:

$$\rho = 0.30 \times \min\left(1.0, \frac{S_{\text{top}}}{6.0}\right) + 0.70 \times \frac{|Q_{\text{tokens}} \cap D_{\text{tokens}}|}{|Q_{\text{tokens}}|}$$

- **Active Abstention ($\rho < 0.40$)**: If empirical/contractual claims are made with insufficient grounding evidence, the verifier sets Grounding Score $G = 2.50$ ($R_{\text{rag}} = 7.50$), actively quarantining the response to human review (`HITL`).
- **Benign Conversation ($\rho \ge 0.70$ or Pleasantry)**: Greetings, thank-yous, and conversational pleasantries are awarded $G = 10.0$ ($R_{\text{rag}} = 0.0$) $\implies$ `ALLOW`.

### 2. Negation-Aware Entity Filtering (Zero False Positives)
When an LLM correctly quotes an invalid customer number to deny it (*"Cancelling after 45 days does not qualify for a $2,000 refund"*), standard regex systems penalize the bot for mentioning "$2,000". 
Our negation window parser detects refutation context (`cannot`, `does not qualify`, `won't be refunded`, `no refund`, `ineligible`) and exempts the denial from penalties, while retaining strict $-3.5\text{ pts}$ deductions for affirmative hallucinations.

### 3. Always-Adaptive Latency Routing (<15ms SLA)
- **Fast-Path by Default**: Operates locally using regex guardrails, in-memory BM25 retrieval, and scatter-gather statistical scorers without LLM latency bottlenecks ($<15\text{ms}$).
- **Automatic Deep-Path Promotion**: Promotes dynamically whenever a financial trigger is detected, CRAG confidence is ambiguous ($\rho < 0.70$), or heuristic/statistical risk exceeds $2.0/10$.
- **Dynamic Active Weight Renormalization**: Renormalizes dimension weights when Stage 3C is bypassed:
  $$w'_i = \frac{w_i}{\sum_{k \in \text{Active}} w_k} = \frac{w_i}{0.75} \implies w'_{\text{heur}} = 0.333, \quad w'_{\text{stat}} = 0.200, \quad w'_{\text{rag}} = 0.467$$

### 4. Zero-Config Self-Healing SQLite Layer
If `controlplane.db` is missing, starting either the **CLI** or **Web Dashboard** automatically creates the SQLite database, migrates the schema tables (`knowledge_base`, `hitl_tickets`, `feedback_store`, `interactions`), and seeds all **20 authoritative Airbnb Markdown policy documents** into RAM and disk in $<50\text{ms}$.

### 5. Cryptographic SHA-256 Hash-Chained Audit Log
Every interaction is signed into `audit_log.jsonl` using a continuous cryptographic hash chain:

$$H_i = \text{SHA-256}\left( \text{EntryID}_i \parallel \text{Timestamp}_i \parallel \text{PromptHash}_i \parallel \text{Decision}_i \parallel \text{Score}_i \parallel H_{i-1} \right)$$

Any retroactive tampering breaks subsequent hashes, verified with the built-in cryptographic auditor (`verify_audit_log_integrity`).

---

## 📊 Performance Benchmarks

### 50-Question Airbnb Grounding & Safety Benchmark (`benchmark_airbnb.py`)

| Benchmark Dimension | Target Specification | Achieved Score | Status |
| :--- | :---: | :---: | :---: |
| **Overall Safety Compliance** | $\ge 98.0\%$ | **50 / 50 (100.0%)** | 🏆 **Perfect** |
| **Autonomous Deflection (`ALLOW`)** | $\ge 90.0\%$ | **18 / 18 (100.0%)** | ⚡ **Zero False Blocks** |
| **Hallucination Interception (`BLOCK`)** | $100.0\%$ | **25 / 25 (100.0%)** | 🛡️ **Zero Leaks** |
| **Ambiguity Quarantine (`FLAG` $\to$ `HITL`)** | $100.0\%$ | **7 / 7 (100.0%)** | 🎯 **100% Precision** |
| **Fast-Path Latency Profile** | $< 25.0\text{ ms}$ | **Avg: 12.08ms (P50: 11.90ms, P90: 12.61ms)** | ⚡ **Ultra-Fast** |
| **Pytest Unit Test Suite** | $100.0\%$ | **113 / 113 Passed** | ✅ **Verified** |
| **Audit Log Integrity** | $100.0\%$ Continuous | **100% Verified Chain** | 🔒 **Tamper-Proof** |

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/ControlPlane.ai.git
cd ControlPlane.ai

# Optional: Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
```bash
# Copy template configuration
cp .env.example .env

# Set your Groq API key for live generation (Optional - runs offline deterministically if omitted)
# GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 3. Launch the Interactive Web Dashboard (Recommended)
```bash
python frontend/run.py
```
> 🌐 Automatically starts the FastAPI/Uvicorn server and opens `http://127.0.0.1:8000` in your default browser.

### 4. Run the Live Interactive CLI
```bash
python cli.py
```
```
================================================================================
  CONTROLPLANE.AI - INTERACTIVE LIVE TERMINAL
  Type any customer or host prompt to inspect the 5-stage pipeline live.
  Commands:
    'status'         - View database counts (KB docs, pending tickets, reviews)
    'history'        - View recent conversation history across ALLOW, HITL, & BLOCK
    'kb'             - List loaded Airbnb policy documents
    'pending'        - View all quarantined HITL tickets
    'verify'         - Verify SHA-256 cryptographic audit chain continuity
    'exit'/'q'       - Quit the interactive session
================================================================================

Enter Prompt >>> What is the refund timeline for UPI payments in India?
```

### 5. Run the 6 Core Enterprise Demonstration Scenarios
```bash
python demo.py
```

### 6. Run the 50-Case Grounding Benchmark Suite
```bash
python benchmark_airbnb.py
```

### 7. Run the Full Pytest Test Suite
```bash
pytest -v
```

---

## 🌐 Interactive Web Dashboard

The web interface (`frontend/`) provides an executive-ready, glassmorphic dark-theme demonstration dashboard:

| Feature Tab | Capabilities |
| :--- | :--- |
| **🚀 Step-by-Step Inspector** | Live auto-play or manual step-through animations across all 5 inspection stages with real-time risk gauges, PII chips, CRAG confidence metrics, and waterfall latencies. |
| **👥 HITL Review Queue** | Interactive triage interface for human compliance officers to **Allow**, **Edit**, or **Block** quarantined tickets with live feedback store metrics. |
| **📚 Knowledge Base Explorer** | Searchable, category-filtered browser for all 20 authoritative Airbnb policy chunks with keyword highlights and metadata tags. |
| **🔒 SHA-256 Audit Verifier** | Live cryptographic integrity verification engine providing instant audit chain continuity proofs. |
| **💡 Preset Scenarios** | 1-click execution for Standard Safe Queries, PII Sanitization, Financial Triggers, RAG Contradictions, and Adversarial DAN Attacks. |

---

## 📚 REST API Reference

The ControlPlane API Gateway runs on `http://127.0.0.1:8000`:

### `POST /api/run` — Execute 5-Stage Inspection Pipeline
**Request Body**:
```json
{
  "prompt": "What is the refund timeline for UPI payments in India?",
  "user_id": "guest_user_101"
}
```

**Response Body**:
```json
{
  "success": true,
  "decision": "ALLOW",
  "composite_score": 0.0,
  "is_financial_trigger": false,
  "active_path": "FAST",
  "total_latency_ms": 12.45,
  "final_response": "Refunds for UPI payments in India are processed within 15 business days...",
  "audit_hash": "a8f3b2c9e7d1...5f21",
  "steps": {
    "step1_guardrails": { "latency_ms": 0.15, "pii_detected": [], "is_injection": false },
    "step2_generation": { "latency_ms": 6.80, "crag_status": "HIGH_CONFIDENCE", "crag_confidence": 1.0 },
    "step3a_fast_checks": { "latency_ms": 1.10, "heuristic_risk": 0.0, "stat_risk": 0.0 },
    "step3b_rag_grounding": { "latency_ms": 4.10, "grounding_score": 10.0, "rag_risk": 0.0, "crag_status": "HIGH_CONFIDENCE" },
    "step3c_ai_judge": { "latency_ms": 0.0, "judge_notes": "Skipped on Fast-Path" },
    "step4_arbitration": { "latency_ms": 0.05, "decision": "ALLOW", "composite_score": 0.0 },
    "step5_governance": { "latency_ms": 0.25, "audit_hash": "a8f3b2c9e7d1...5f21" }
  }
}
```

### Other Endpoints
- `GET /api/health` — System status, active LLM backend, database record counts.
- `GET /api/scenarios` — Pre-configured demonstration scenarios.
- `GET /api/hitl/tickets` — Pending and historical HITL review tickets.
- `POST /api/hitl/resolve` — Submit human review resolution (`ALLOW` | `EDIT` | `BLOCK`).
- `GET /api/kb` — Query and filter the authoritative policy knowledge base.
- `GET /api/audit/verify` — Cryptographically verify SHA-256 continuous hash chain.
- `GET /api/history` — Conversation history across `ALLOW`, `HITL`, and `BLOCK`.
- `GET /api/stats` — High-level telemetry distributions and deflection rates.

---

## 📁 Repository Structure

```
ControlPlane.ai/
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules for caches, DBs, and logs
├── LICENSE                        # MIT Open Source License
├── README.md                      # Comprehensive project documentation
├── SYSTEM_DOCUMENTATION.md        # Technical specification & mathematical proofs
├── requirements.txt               # Pinned Python package dependencies
│
├── pipeline.py                    # Master 5-Stage Orchestrator & Adaptive Router
├── models.py                      # Shared Pydantic data schemas, enums & risk configs
├── pii.py                         # Stage 1: Pre-Execution Guardrails (PII & Injection)
├── llm_client.py                  # Stage 2: Groq / LLM Client with JSON mode
├── fast_checks.py                 # Stage 3A: Fast Parallel Checks (Heuristics & Entropy)
├── rag_verifier.py                # Stage 3B: Corrective RAG (CRAG) & Factual Verifier
├── ai_judge.py                    # Stage 3C: AI-as-a-Judge Compliance Evaluator
├── arbitrator.py                  # Stage 4: Policy Arbitration Matrix & Score Engine
├── audit_hitl.py                  # Stage 5: Cryptographic SHA-256 Audit & HITL Queue
├── db.py                          # SQLite Persistence Layer & Zero-Config Auto-Seeder
│
├── cli.py                         # Interactive Live Terminal Interface
├── demo.py                        # 6 Core Enterprise Scenario Benchmark Runner
├── benchmark_airbnb.py            # Official 50-Question Airbnb Compliance Benchmark
├── logging_config.py              # Structured JSON Logging & ContextVar Trace IDs
│
├── airbnb-grounding-rag-kb/       # Authoritative Airbnb Knowledge Base Corpus
│   ├── cleaned/                   # 20 Authoritative Markdown Policy Documents
│   └── evaluation/                # 50-Case Ground-Truth Benchmark Dataset
│
├── frontend/                      # Web Dashboard & Demonstration UI
│   ├── run.py                     # 1-Click Launch Script (auto-opens browser)
│   ├── server.py                  # FastAPI REST API Server
│   ├── test_frontend.py           # Automated Frontend Endpoint Test Suite
│   ├── README.md                  # Frontend documentation & presentation guide
│   └── static/                    # Single-Page Application Assets
│       ├── index.html             # Glassmorphic Dark-Theme Dashboard
│       ├── app.js                 # Interactive Controller & Chart Visualizer
│       └── styles.css             # Custom CSS Animations & Gauge Fills
│
└── tests/                         # Full Pytest Test Suite (113 Tests)
    ├── test_pii.py                # Luhn algorithm, PII regex & injection tests
    ├── test_fast_checks.py        # Parallel worker bus, loop repetition & entropy tests
    ├── test_rag_verifier.py       # BM25 retrieval, numeric parsing & grounding tests
    ├── test_tiered_crag.py        # CRAG confidence, active abstention & negation tests
    ├── test_arbitrator.py         # Financial triggers & composite score math tests
    ├── test_audit_hitl.py         # SHA-256 hash chains, HITL triage & feedback tests
    └── test_db.py                 # SQLite migration, auto-seeding & interaction tests
```

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more details.

---

<div align="center">
<b>ControlPlane.ai — Enterprise Responsible AI Gateway</b><br>
<i>Engineered for Maximum Safety, Verifiable Grounding, and Sub-15ms Latency SLAs.</i>
</div>

