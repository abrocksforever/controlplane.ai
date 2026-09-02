# ControlPlane.ai — Responsible AI Control Plane & Evaluative Gateway

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pytest](https://img.shields.io/badge/Tests-113%2F113%20Passed%20(100%25)-059669?logo=pytest&logoColor=white)](tests/)
[![Benchmark](https://img.shields.io/badge/Benchmark-50%2F50%20Compliant%20(100%25)-6366F1)](benchmark_airbnb.py)
[![Fast-Path Latency](https://img.shields.io/badge/Fast--Path%20Latency-%3C%2015ms%20(P50%3A%2011.9ms)-06B6D4)](#performance-benchmarks)
[![Audit Log](https://img.shields.io/badge/Audit%20Log-SHA--256%20Hash%20Chained-8B5CF6)](#5-cryptographic-sha-256-hash-chained-audit-log)
[![License](https://img.shields.io/badge/License-MIT-F59E0B.svg)](LICENSE)

**A high-performance, modular Responsible AI Control Plane featuring sub-millisecond pre-execution guardrails, parallel scatter-gather heuristics, Corrective RAG (CRAG) factual grounding, context-aware sliding negation filtering, AI-as-a-Judge compliance evaluation, 3-tier policy arbitration, and immutable cryptographic SHA-256 hash-chained audit logging.**

[Quick Start](#quick-start) • [Web Dashboard Guide](#interactive-web-dashboard-guide) • [Key Innovations](#key-architectural-innovations) • [Benchmark Scorecard](#performance-benchmarks) • [API Reference](#rest-api-reference) • [Directory Structure](#repository-structure)

</div>

---

## Problem Statement Overview (PS1 Alignment)

Enterprise deployment of Large Language Models introduces severe operational, financial, and regulatory vulnerabilities:
- **Sensitive Data Exfiltration**: Accidental disclosure of customer PII (SSNs, credit cards, emails, phone numbers, secret API keys).
- **Adversarial Jailbreaks**: Prompt injection, developer-mode bypasses, and "Do Anything Now" (DAN) attacks overriding system instructions.
- **Hallucinations & Unauthorized Commitments**: LLMs fabricating refund amounts, policy timelines, or unauthorized financial payouts.
- **Regulatory Non-Compliance**: Lack of immutable, tamper-proof audit trails required by frameworks such as EU AI Act, NIST AI RMF, and ISO 42001.

Traditional AI guardrail architectures create a painful trade-off:
1. **Monolithic LLM Judges**: Introducing 2,000–4,000ms latency penalties on every single turn, inflating API inference costs and degrading user experience.
2. **Naive Keyword Filters**: Triggering brittle false positives that block legitimate customer service denials (e.g., penalizing an LLM for quoting a denied amount when saying *"You are not eligible for a $2,000 refund"*).

**`ControlPlane.ai`** resolves this dilemma through a decoupled, multi-tiered evaluative control plane designed to:
1. **Intercept Threats Early ($<1\text{ms}$)**: Redact PII in-flight with reverse-offset slicing, validate credit cards via Luhn checksums, and terminate adversarial jailbreaks pre-execution.
2. **Eliminate Hallucinations with Corrective RAG (CRAG)**: Compute normalized retrieval quality ($\rho \in [0, 1]$), actively abstain on unindexed empirical claims, and apply 60-character sliding negation-window parsing to eliminate false positive refutations.
3. **Route Adaptively for Real-Time SLAs**: Deliver clean routine inquiries in **$<15\text{ms}$** (Fast-Path), dynamically auto-elevating to Deep-Path ($<3000\text{ms}$) only upon financial triggers, CRAG retrieval ambiguity, or statistical risk spikes.
4. **Enforce Absolute Auditability**: Cryptographically chain every prompt, decision, risk score, and timestamp into an immutable SHA-256 ledger with an integrated Human-in-the-Loop (HITL) triage queue.

---

## 5-Stage Modular Pipeline Architecture

```
                                  [ User Prompt ]
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STAGE 1: Pre-Execution Guardrails (pii.py) [<1.0 ms]                        │
  │ • Reverse-Offset Slice Masking (SSN, Email, Phone, API Keys)                │
  │ • Luhn Algorithmic Checksum Validation (Visa, Mastercard, Amex, Discover)   │
  │ • Weighted Adversarial Prompt Injection Classifier (DAN / Jailbreak Filter) │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │ (Sanitized / PII-Masked Prompt)
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STAGE 2: Context Retrieval & Primary Generation (llm_client.py)             │
  │ • In-Memory BM25 Knowledge Retrieval (<0.05ms) + Authoritative Airbnb Corpus │
  │ • Corrective RAG (CRAG) Retrieval Quality Evaluator (ρ calculation)         │
  │ • Primary LLM Generation with Token Budgeting (Groq / Qwen / Llama-3)       │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │ (Candidate Response Draft)
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
  │ FAST-PATH CONCURRENT CHECKS  │                │ DEEP-PATH SCATTER-GATHER BUS │
  │ (<15ms SLA)                  │                │ (<1.0s Wall-Clock Latency)   │
  │                              │                │                              │
  │ [STAGE 3A: Fast Checks]      │                │ [STAGE 3A: Fast Checks]      │
  │ • Scatter-Gather Worker Bus  │                │ • Output PII & Banned Lexicon│
  │ • Output PII & Banned Lexicon│                │ • Shannon Entropy & Loops    │
  │ • Shannon Entropy & Loops    │                │                              │
  │                              │                │ [STAGE 3B: Batched NLI]      │
  │ [STAGE 3B: Negation Filter]  │                │ • Single-Call Batched NLI    │
  │ • Numeric Span Extraction    │                │ • Assertion Pre-Filtering    │
  │ • Sliding Negation Window    │                │ • 60-Char Negation Parsing   │
  │ • In-Memory BM25 Grounding   │                │                              │
  │                              │                │ [STAGE 3C: AI-as-a-Judge]    │
  │                              │                │ • Demographic Bias Detection │
  │                              │                │ • Hostile Tone Evaluation    │
  │                              │                │ • Token Budgeted (max: 400)  │
  └──────────────┬───────────────┘                └──────────────┬───────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │ (Risk Dimensions Breakdown)
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STAGE 4: Policy Arbitration & Risk Assessment (arbitrator.py)               │
  │ • Dynamic Active Weight Renormalization (w' = w_i / Σ w_active)             │
  │ • Financial Gate Trigger (FinCheck: Wire Transfers / Bank Info / Payouts)   │
  │ • 3-Tier Policy Decision Matrix:                                            │
  │     [ALLOW: S ≤ 2.50]  │  [HITL Quarantine: 2.50 < S < 7.00]  │  [BLOCK: S ≥ 7.00]
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STAGE 5: Governance, HITL Queue & Cryptographic Audit (audit_hitl.py)       │
  │ • Persistent SQLite HITL Triage Queue (ALLOW / EDIT / BLOCK Review)         │
  │ • Active Learning Continuous Feedback Store                                 │
  │ • Immutable O(1) Reverse-Seek SHA-256 Hash Chained Audit Ledger             │
  └─────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Stage Breakdown

| Stage | Primary Module | Core Functionality | Latency SLA |
| :--- | :--- | :--- | :--- |
| **Stage 1: Pre-Execution Guardrails** | `pii.py` | Reverse-offset PII redaction, Luhn validation, weighted prompt injection classification | $<1.0\text{ ms}$ |
| **Stage 2: Retrieval & Generation** | `llm_client.py`, `db.py` | In-memory BM25 retrieval across 20 Markdown policies, CRAG confidence score ($\rho$), draft generation with token budgeting (`max_tokens=250`) | $<0.05\text{ ms}$ (Ret) |
| **Stage 3A: Fast Parallel Checks** | `fast_checks.py` | Scatter-gather worker bus, output PII, banned lexicon, Shannon entropy, degenerate loops | $<1.0\text{ ms}$ |
| **Stage 3B: Batched Factual Grounding** | `rag_verifier.py` | Single-call batched NLI claim entailment, assertion pre-filtering, 60-char sliding negation window parsing, active abstention | $<800\text{ ms}$ (Deep) / $0.0\text{ ms}$ (Fast) |
| **Stage 3C: AI-as-a-Judge** | `ai_judge.py` | Concurrent compliance evaluation via Groq JSON mode with token budgeting (`max_tokens=400`) for bias, tone, and policy | $<1000\text{ ms}$ (Deep only) |
| **Stage 3 Parallel Bus (Wall-Clock)** | `pipeline.py` | Scatter-Gather concurrency via `ThreadPoolExecutor(max_workers=3)`: $\max(T_{3A}, T_{3B}, T_{3C})$ | $<1.0\text{ s}$ (Deep) / $<13\text{ ms}$ (Fast) |
| **Stage 4: Policy Arbitration** | `arbitrator.py` | Composite risk score calculation ($S \in [0, 10]$), financial trigger gate, 3-tier routing | $<0.1\text{ ms}$ |
| **Stage 5: Audit & Governance** | `audit_hitl.py` | Cryptographic SHA-256 recursive hash chaining, SQLite HITL ticket quarantine, active feedback | $<15.0\text{ ms}$ |

---

## Key Architectural Innovations

### 1. Corrective RAG (CRAG) & Active Abstention Gate
Standard RAG systems blindly forward ungrounded candidate claims when relevant documents cannot be found. ControlPlane.ai computes a normalized retrieval confidence score $\rho \in [0.0, 1.0]$ combining BM25 top-rank relevance and token coverage:

$$\rho = 0.30 \times \min\left(1.0, \frac{S_{\text{top}}}{6.0}\right) + 0.70 \times \frac{|Q_{\text{tokens}} \cap D_{\text{tokens}}|}{|Q_{\text{tokens}}|}$$

- **High Confidence ($\rho \ge 0.70$)**: Sufficient authoritative grounding is present; candidate proceeds along the Fast-Path.
- **Ambiguous ($0.40 \le \rho < 0.70$)**: Retrieval density is marginal; pipeline auto-promotes to Deep-Path Semantic NLI and secondary AI Judge verification.
- **Knowledge Gap Active Abstention ($\rho < 0.40$)**: If empirical or contractual claims are asserted without indexed grounding evidence, the verifier sets Grounding Score $G = 2.50$ ($R_{\text{rag}} = 7.50$), actively quarantining the response to human review (`HITL`).
- **Benign Pleasantry Exemption**: Conversational greetings and expressions of gratitude without factual claims are assigned $G = 10.0$ ($R_{\text{rag}} = 0.0$) $\implies$ `ALLOW`.

### 2. Negation-Aware Entity Filtering (Minimizing False Positive Refutations)
When a customer asks *"Can I cancel after 45 days for a $2,000 refund?"*, an accurate AI assistant must quote the user's invalid number to deny it: *"Cancelling after 45 days does not qualify for a $2,000 refund."*

Standard regex and NER guardrails flag the presence of "$2,000" as an ungrounded hallucination. Our **60-character sliding negation-window parser** inspects the preceding and succeeding tokens for refutation markers (`cannot`, `does not qualify`, `won't be refunded`, `no refund`, `ineligible`, `not eligible`).
- **Refutation Match**: The span is recognized as a valid denial and exempted from penalty (avoiding false positive penalties on legitimate customer service denials).
- **Affirmative Hallucination**: Ungrounded numeric assertions (e.g., *"You will receive a $2,000 refund"*) receive a strict $-3.5\text{ pt}$ deduction per violation.

### 3. Always-Adaptive Latency Routing (<15ms SLA)
- **Fast-Path by Default ($<15\text{ms}$)**: Routine inquiries are processed locally using pre-compiled regex guardrails, in-memory BM25 retrieval, and scatter-gather statistical scorers—eliminating secondary LLM latency bottlenecks.
- **Dynamic Deep-Path Elevation**: Traffic is automatically promoted to full Semantic NLI and secondary AI-as-a-Judge inspection if:
  1. A financial trigger keyword is detected (`wire transfer`, `bank account`, `payout`, amounts $\ge \$1,000$).
  2. CRAG retrieval confidence is ambiguous ($0.40 \le \rho < 0.70$).
  3. Heuristic or statistical anomaly score exceeds threshold ($R_{\text{heur}} > 2.0$ or $R_{\text{stat}} > 2.0$).
- **Dynamic Active Weight Renormalization**: When Stage 3C is bypassed on the Fast-Path, active dimension weights are dynamically renormalized:
  $$w'_i = \frac{w_i}{\sum_{k \in \text{Active}} w_k} = \frac{w_i}{0.75} \implies w'_{\text{heur}} = 0.333, \quad w'_{\text{stat}} = 0.200, \quad w'_{\text{rag}} = 0.467$$
- **3-Tier Decision Matrix**:
  - **ALLOW** ($S \le 2.50$): Delivered directly to client.
  - **HITL Quarantine** ($2.50 < S < 7.00$ or Financial Trigger): Quarantined in SQLite review queue.
  - **BLOCK** ($S \ge 7.00$ or Critical Stage 1 Injection): Terminated with safe canned fallback.

### 4. Scatter-Gather Concurrent Parallel Bus (Stage 3A, 3B, 3C)
Previously, post-generation compliance checks ran in a sequential waterfall ($T_{\text{stage3}} = T_{3A} + T_{3B} + T_{3C}$), causing total turnaround times to exceed 30 seconds on complex prompts. ControlPlane.ai now dispatches all three inspection gates concurrently via Python's `ThreadPoolExecutor(max_workers=3)`:
- **Worker 1 (Stage 3A)**: Evaluates output PII, banned lexicon, Shannon entropy, and loop repetition in parallel memory ($<1\text{ms}$).
- **Worker 2 (Stage 3B)**: Runs BM25 grounding and batched NLI claim entailment.
- **Worker 3 (Stage 3C)**: Runs AI-as-a-Judge compliance evaluation via Groq JSON mode with token budgeting.

Total Stage 3 Wall-Clock Latency is now bounded by the maximum single worker:
$$T_{\text{parallel}} = \max(T_{3A}, T_{3B}, T_{3C}) \approx 990\text{ ms} \quad (\text{vs. } 32,069\text{ ms sequential})$$

### 5. Single-Call Batched NLI & Conversational Assertion Pre-Filtering
Standard sentence-level NLI verifiers iterate over candidate sentences in a serial loop (`for sentence in sentences: call_llm(...)`), triggering multiple sequential round-trips to remote LLMs. ControlPlane.ai introduces a dual-layer optimization:
1. **Conversational Assertion Pre-Filtering (`evaluate_factual_assertions`)**: Scans candidate sentences for empirical assertions (currency symbols, numeric quantities, timeframes, policy actions). Polite conversational pleasantries (e.g., *"Have a safe trip!"*, *"Thank you for contacting us"*) are recognized in $0.01\text{ms}$ and exempted from remote NLI evaluation.
2. **Single-Call Batched NLI (`_run_batch_nli_entailment`)**: All candidate empirical claims requiring verification are bundled into a single structured prompt and evaluated in **one single batched JSON call**:
   ```json
   {
     "results": [
       {"index": 1, "label": "entailed"},
       {"index": 2, "label": "unsupported"}
     ]
   }
   ```
   This slashes Stage 3B latency from **26,677 ms down to <800 ms** (or $0.0\text{ms}$ when local heuristics suffice).

### 6. Primary Generation & Judge Token Budgeting
- **Primary LLM Generation (Stage 2)**: Constrained with `max_tokens=250` budget to prevent rambling while answering customer queries thoroughly, cutting generation time from 2,311ms to ~1,500ms.
- **AI-as-a-Judge (Stage 3C)**: Budgeted with `max_tokens=400` and powered by high-throughput evaluation models (`openai/gpt-oss-20b` or `CONTROLPLANE_EVAL_MODEL`), dropping evaluation latency from 5,392ms down to **~985 ms**.

### 7. Zero-Config Self-Healing SQLite Layer
If `controlplane.db` is missing upon startup, launching the **Interactive Web Dashboard** automatically:
1. Initializes all SQLite schema tables (`knowledge_base`, `hitl_tickets`, `feedback_store`, `interactions`).
2. Discovers and indexes all **20 authoritative Airbnb Markdown policy documents** from `airbnb-grounding-rag-kb/cleaned/`.
3. Warms the in-memory BM25 index in $<50\text{ms}$ with zero manual migration or configuration required.

### 8. Cryptographic SHA-256 Hash-Chained Audit Log
Every pipeline execution produces a signed, immutable record in `audit_log.jsonl`. Each entry is linked to its predecessor via recursive cryptographic hashing:

$$H_i = \text{SHA-256}\left( \text{EntryID}_i \parallel \text{Timestamp}_i \parallel \text{PromptHash}_i \parallel \text{Decision}_i \parallel \text{Score}_i \parallel H_{i-1} \right)$$

- Genesis entry ($i = 0$) anchors the chain with $H_0 = \text{"0"} \times 64$.
- Any retroactive alteration or deletion breaks all subsequent hashes in the chain.
- The built-in cryptographic auditor (`verify_audit_log_integrity`) performs $O(1)$ reverse-seek verification to validate full ledger integrity.

---

## Performance Benchmarks

### 1. End-to-End Latency Profile & Parallel Bus Benchmark

Latency measurements differentiate between **ControlPlane.ai Gateway evaluation overhead** and **downstream Primary LLM generation time** over external cloud APIs.

**Test Scenario**: High-liability financial query evaluated on Deep-Path (*"I was double charged and need a $2,000 wire transfer payout immediately"*):

| Pipeline Component | Sequential Baseline (P50) | Concurrent Parallel Bus (P50) | Execution Type | Operational Notes |
| :--- | :---: | :---: | :--- | :--- |
| **Stage 1: Pre-Execution Guardrails** | $0.8\text{ ms}$ | **$0.1\text{ ms}$** | Local CPU | In-memory regex & Luhn checksum |
| **Stage 2: Primary LLM Generation** | $2,311\text{ ms}$ | **$1,910\text{ ms}$** | Cloud API (WAN) | Token-budgeted at 250 max tokens |
| **Stage 3A: Fast Parallel Checks** | $5.0\text{ ms}$ | **$0.2\text{ ms}$** | Local CPU (Parallel) | In-memory entropy & lexicon scanner |
| **Stage 3B: RAG Grounding Verification** | $26,677\text{ ms}$ | **$0.0\text{ - }780\text{ ms}$** | Hybrid Local / API | $0.0\text{ ms}$ when assertion-filtered; ~780ms when batched NLI invoked |
| **Stage 3C: AI-as-a-Judge Compliance** | $5,392\text{ ms}$ | **$985\text{ ms}$** | Cloud API (Parallel) | Structured JSON compliance evaluation (`max_tokens=400`) |
| ──────────────────────────────────── | ──────────── | ──────────── | ─────────────────── | ────────────────────────────────────────────── |
| **Stage 3 Wall-Clock Bus Latency** | $32,069\text{ ms}$ | **$999\text{ ms}$** | Multi-Threaded Bus | $\max(T_{3A}, T_{3B}, T_{3C})$ concurrent execution |
| **Stage 4: Policy Arbitration** | $0.1\text{ ms}$ | **$0.1\text{ ms}$** | Local CPU | Mathematical weight normalization |
| **Stage 5: Cryptographic Governance** | $23.4\text{ ms}$ | **$12.4\text{ ms}$** | Local Disk I/O | SQLite write + SHA-256 hash chaining |
| ==================================== | ============ | ============ | =================== | ============================================== |
| **Total Turnaround (Deep-Path)** | **$34,408\text{ ms}$** | **$2,941\text{ ms}$** | End-to-End System | **91.5% Latency Reduction** (~34.4s to ~2.9s) |
| **Total Turnaround (Fast-Path Routine)**| **$\sim 2,350\text{ ms}$** | **$1,512\text{ ms}$** | End-to-End System | **$<13\text{ ms}$ Total Gateway Inspection Overhead** |

*Environment: Windows 11 64-bit, Python 3.13, local in-memory BM25 + SQLite 3.45. Remote inference: Groq Cloud LPU endpoints (`qwen/qwen3.8-27b` for primary generation, `openai/gpt-oss-20b` for evaluation).*

---

### 2. 50-Case Grounding & Compliance Benchmark (`benchmark_airbnb.py`)

The benchmark evaluates 50 curated Airbnb domain scenarios spanning clean policy queries, ungrounded traps, cancellation disputes, and financial trigger edge cases:

| Metric | Measured Result | Benchmark Standard | Operational Meaning |
| :--- | :---: | :---: | :--- |
| **Evaluated Test Cases** | **50 / 50** | 50 Cases | 18 factual FAQs, 25 policy violations, 7 edge cases |
| **Safety Containment Rate** | **50 / 50 (100.0%)** | $\ge 98.0\%$ | Zero unauthorized or ungrounded responses reached client unvetted |
| **Autonomous Clean Deflection (`ALLOW`)** | **18 / 18 (100.0%)** | $\ge 90.0\%$ | Valid in-scope policy FAQs delivered without human review |
| **Direct Hard Gateway Rejections (`BLOCK`)** | **4 / 25 (16.0%)** | Tracked | Immediate termination for critical prompt injection and explicit contradictions |
| **Quarantine Escalation to Triage (`HITL`)** | **21 / 25 (84.0%)** | Tracked | High-risk and ambiguous claims routed to human queue ($2.5 < S < 7.0$) |
| **Ambiguity Triage Capture (`FLAG` $\to$ `HITL`)** | **7 / 7 (100.0%)** | $\ge 95.0\%$ | Edge cases with partial retrieval grounding routed to human review |
| **False Rejection Rate on Clean FAQs** | **0 / 18 (0.0%)** | $\le 5.0\%$ | Zero legitimate policy answers falsely blocked on benchmark suite |
| **Fast-Path Gateway Overhead (P50)** | **11.84 ms** | $< 15.0\text{ ms}$ | Evaluative gateway latency on routine traffic |
| **Fast-Path Gateway Overhead (P90)** | **12.61 ms** | $< 20.0\text{ ms}$ | 90th percentile evaluative gateway latency |
| **Cryptographic Ledger Continuity** | **100% Verified** | Continuous | Complete SHA-256 hash continuity verified via reverse-seek |
| **Pytest Automated Test Coverage** | **113 / 113 Passed** | 100% Pass | Unit and integration test suite passing |

---

## Quick Start

### 1. Clone & Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/ControlPlane.ai.git
cd ControlPlane.ai

# Create and activate virtual environment
# On Linux / macOS:
python -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

```bash
# Copy template configuration
cp .env.example .env

# Optional: Add Groq API Key for live online LLM generation
# If omitted, ControlPlane.ai operates in deterministic offline mock mode
# GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 3. Launch the Interactive Web Dashboard

Launch the application with a single command:

```bash
python frontend/run.py
```

>  **Automatic Browser Launch**: This starts the FastAPI / Uvicorn server on `http://127.0.0.1:8000`, initializes the SQLite database with all 20 authoritative policy documents, and automatically opens the interactive dashboard in your default browser.
> 
> *Alternatively, start via Uvicorn directly:*
> ```bash
> python frontend/server.py --port 8000
> ```

### 4. Running Benchmarks & Verification (Optional)

```bash
# Run the 50-case Airbnb Grounding Benchmark Suite
python benchmark_airbnb.py

# Run the complete Pytest test suite (113 tests)
pytest -v
```

---

## Interactive Web Dashboard Guide

The web interface (`frontend/`) is an executive-ready, glassmorphic Single-Page Application (SPA) designed for interactive step-by-step demonstration, real-time risk inspection, and human compliance operations.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  ControlPlane.ai   [Live Demonstration] [HITL Queue (3)] [Knowledge Base] [Audit Log] [Theme] │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  PRESET SCENARIOS:                                                                        │
│ [ 1. Safe UPI Inquiry ]  [ 2. Financial $3.5k Payout ]  [ 3. Credit Card PII ]            │
│ [ 4. Ungrounded Refund Trap ]  [ 5. DAN Jailbreak Attack ]  [ 6. Fabricated Guarantee ]     │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  PROMPT INPUT & CONTROLS:                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  Execution: [ADAPTIVE ▼]      │
│  │ What is the refund timeline for UPI payments in India?   │  [Run Full Pipeline]        │
│  └──────────────────────────────────────────────────────────┘  [Step-by-Step Mode]       │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  STEP-BY-STEP INSPECTION WATERFALL:                                                         │
│  [● Stage 1: Guardrails] ──► [● Stage 2: CRAG] ──► [● Stage 3: Checks] ──► [● Stage 4: Arb] │
│                                                                                             │
│  ┌─────────────────────────┐ ┌──────────────────────────┐ ┌───────────────────────────────┐ │
│  │ Stage 1: Pre-Guardrails │ │ Stage 2: CRAG Grounding  │ │ Stage 4: Policy Arbitration   │ │
│  │ PII Detected: None      │ │ Retrieval: ρ = 1.00 (High│ │ Decision: [ ALLOW ]            │ │
│  │ Injection Risk: 0.0/10  │ │ Top Doc: KB-AIRBNB-004   │ │ Composite Risk Score: 0.00/10 │ │
│  │ Latency: 0.15ms         │ │ Latency: 6.80ms          │ │ Latency: 0.05ms               │ │
│  └─────────────────────────┘ └──────────────────────────┘ └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Live Demonstration & Stepper Mode (`/static/index.html`)

- **Interactive Stepper vs Full Run**:
  - **Full Run**: Click **"Run Full Pipeline"** for immediate end-to-end evaluation with real-time waterfall timing.
  - **Step-by-Step Walkthrough**: Toggle **"Step-by-Step Mode"** to pause at each stage. Use **"Next Step >"** or **"Auto Play"** to observe stage-by-stage transformations (sanitization $\to$ retrieval $\to$ parallel scoring $\to$ arbitration $\to$ audit signing).
- **Execution Mode Selector**:
  - **ADAPTIVE (Default)**: Automatically routes clean queries to Fast-Path ($<15\text{ms}$) and auto-promotes to Deep-Path on ambiguity or risk triggers.
  - **FAST**: Bypasses secondary LLM judging for maximum throughput.
  - **DEEP**: Forces full Semantic NLI and secondary AI-as-a-Judge inspection.
- **Stage-by-Stage Inspector Cards**:
  - **Stage 1 (Pre-Guardrails)**: Displays raw prompt, reverse-offset sanitized prompt, detected PII chips (SSN, Cards, Email), and weighted prompt injection gauge.
  - **Stage 2 (Context & CRAG)**: Displays BM25 retrieved policy chunks, token overlap ratio, CRAG quality score $\rho$, and candidate response draft.
  - **Stage 3A (Fast Checks)**: Displays output PII scanner results, banned lexicon hits, Shannon entropy score, and degenerate repetition metrics.
  - **Stage 3B (Factual Grounding)**: Displays extracted numeric/temporal entities, 60-character sliding negation-window matches (exempting denials), and final Grounding Score $G$.
  - **Stage 3C (AI-as-a-Judge)**: Displays bias, tone, and policy adherence scores with structured reasoning from the compliance judge LLM.
  - **Stage 4 (Arbitration)**: Shows active weight breakdown, financial trigger flag, composite score gauge, and the 3-tier routing verdict (`ALLOW`, `HITL`, `BLOCK`).
  - **Stage 5 (Audit & Delivery)**: Displays the delivered text payload, total end-to-end latency, waterfall breakdown, and cryptographic SHA-256 hash stamp.

### 2. One-Click Demonstration Scenario Presets

The dashboard includes **6 built-in enterprise presets** that populate prompts and demonstrate distinct pipeline capabilities with a single click:

| Preset Scenario | Category | Expected Verdict | What It Demonstrates |
| :--- | :--- | :---: | :--- |
| **1. Standard Safe Query** | Routine Inquiry | `ALLOW` ($S = 0.00$) | High CRAG confidence ($\rho = 1.0$), fast-path deflection in $<15\text{ms}$. |
| **2. Financial Payout Trigger** | Financial Risk | `HITL` (Forced) | `FinCheck` detects $3,500 security deposit payout and routes directly to HITL queue. |
| **3. Guest PII Sanitization** | Data Privacy | In-Flight Redaction | Credit card (Luhn-validated) and email masked pre-execution; safe processing. |
| **4. Ungrounded Refund Trap** | Hallucination | `HITL` / `BLOCK` | User asks for a non-existent $2,000 refund; negation filter refutes invalid claim. |
| **5. Adversarial DAN Attack** | Jailbreak | `BLOCK` (Stage 1) | Prompt injection classifier detects "developer mode" override; early termination in $<1\text{ms}$. |
| **6. Fabricated Guarantee** | Policy Mismatch | `HITL` ($R_{\text{rag}} = 7.50$) | RAG verifier catches unindexed empirical guarantee; active abstention triggered. |

### 3. Human-in-the-Loop (HITL) Triage Queue

Navigate to the **HITL Queue** tab to access the live compliance review interface:
- **Pending Ticket Roster**: Lists quarantined tickets with timestamp, requesting user, composite score, and trigger reason.
- **Reviewer Action Modal**:
  - **ALLOW**: Approve and release the original candidate response unchanged.
  - **EDIT**: Open a live text editor to sanitize or correct the candidate response and deliver the human-approved text.
  - **BLOCK**: Override and deliver the safe canned fallback response.
  - **Reviewer Notes**: Record compliance rationale stored into SQLite for active learning.
- **Active Learning Continuous Feedback Metrics**:
  - Displays real-time charts: Total Reviewed, Approval Rate (%), Edit Rate (%), and Block Rate (%).
  - Provides dynamic threshold calibration feedback based on reviewer behavior.

### 4. Authoritative Knowledge Base Explorer

Navigate to the **Knowledge Base** tab to explore the underlying grounding corpus:
- **Interactive Search**: Real-time client-side search across all 20 Markdown policy documents.
- **Category Filtering**: Filter by *Refunds & Cancellations*, *Major Disruptive Events*, *Host Damage Protection*, *Safety & Security*, and *Payment Terms*.
- **Document Detail View**: Inspect full document text, extracted keyword tokens, and metadata tags used by the BM25 retrieval engine.

### 5. Cryptographic SHA-256 Audit Log Verifier

Navigate to the **SHA-256 Audit Log** tab to inspect the tamper-proof ledger:
- **Ledger Explorer**: Scroll through chronological audit entries showing Entry ID, Timestamp, Prompt Hash, Decision, Score, and Hash ($H_i$).
- **One-Click Verification**: Click **"Verify Audit Log Integrity"** to execute a live cryptographic check that re-hashes every entry from genesis to the latest record, confirming continuous unbroken SHA-256 chaining.
- **Payload Inspection**: Click any entry to view its complete JSON telemetry payload.

### 6. Theme Customization

- Switch between **Dark Mode** (glassmorphic cyber-navy) and **Bright Mode** (crisp modern enterprise white) using the sun/moon toggle in the top header.

---

## REST API Reference

The ControlPlane API Gateway runs on `http://127.0.0.1:8000`:

### `POST /api/run` — Execute 5-Stage Inspection Pipeline

**Example cURL Request**:
```bash
curl -X POST http://127.0.0.1:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the refund timeline for UPI payments in India?",
    "user_id": "guest_user_101",
    "execution_mode": "ADAPTIVE"
  }'
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
  "audit_hash": "a8f3b2c9e7d14589f012bb4cd78e90a3451e6789fabc0123456789abcdef5f21",
  "steps": {
    "step1_guardrails": {
      "latency_ms": 0.15,
      "pii_detected": [],
      "is_injection": false,
      "sanitized_prompt": "What is the refund timeline for UPI payments in India?"
    },
    "step2_generation": {
      "latency_ms": 6.80,
      "crag_status": "HIGH_CONFIDENCE",
      "crag_confidence": 1.0
    },
    "step3a_fast_checks": {
      "latency_ms": 1.10,
      "heuristic_risk": 0.0,
      "stat_risk": 0.0,
      "output_pii": []
    },
    "step3b_rag_grounding": {
      "latency_ms": 4.10,
      "grounding_score": 10.0,
      "rag_risk": 0.0,
      "crag_status": "HIGH_CONFIDENCE"
    },
    "step3c_ai_judge": {
      "latency_ms": 0.0,
      "judge_notes": "Skipped on Fast-Path (Clean heuristics & high CRAG confidence)"
    },
    "step4_arbitration": {
      "latency_ms": 0.05,
      "decision": "ALLOW",
      "composite_score": 0.0
    },
    "step5_governance": {
      "latency_ms": 0.25,
      "audit_hash": "a8f3b2c9e7d1...5f21"
    }
  }
}
```

### Other Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health, active LLM backend status, and database record telemetry. |
| `GET` | `/api/scenarios` | Returns pre-configured enterprise demonstration scenario presets. |
| `GET` | `/api/hitl/tickets` | Returns list of pending and resolved Human-in-the-Loop review tickets. |
| `POST` | `/api/hitl/resolve` | Resolves a quarantined ticket (`ALLOW`, `EDIT`, `BLOCK`) with reviewer notes. |
| `GET` | `/api/kb` | Queries and filters the authoritative 20-document policy knowledge base. |
| `GET` | `/api/audit/verify` | Cryptographically verifies continuous SHA-256 audit hash chain integrity. |
| `GET` | `/api/history` | Retrieves past interaction logs with stage telemetry across all decision tiers. |
| `GET` | `/api/stats` | Returns aggregate metrics: deflection rate, HITL queue size, and latency percentiles. |

---

## Repository Structure

```
ControlPlane.ai/
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules for caches, DBs, and logs
├── LICENSE                        # MIT Open Source License
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Pinned Python package dependencies
│
├── pipeline.py                    # Master 5-Stage Orchestrator & Adaptive Router
├── models.py                      # Shared Pydantic data schemas, enums & risk configs
├── pii.py                         # Stage 1: Pre-Execution Guardrails (PII & Injection)
├── llm_client.py                  # Stage 2: Groq / LLM Client with JSON mode & token budgeting
├── fast_checks.py                 # Stage 3A: Fast Parallel Checks (Heuristics & Entropy)
├── rag_verifier.py                # Stage 3B: Corrective RAG (CRAG) & Batched NLI Verifier
├── ai_judge.py                    # Stage 3C: AI-as-a-Judge Parallel Compliance Evaluator
├── arbitrator.py                  # Stage 4: Policy Arbitration Matrix & Score Engine
├── audit_hitl.py                  # Stage 5: Cryptographic SHA-256 Audit & HITL Queue
├── db.py                          # SQLite Persistence Layer & Zero-Config Auto-Seeder
├── logging_config.py              # Structured JSON Logging & ContextVar Trace IDs
├── cli.py                         # Interactive Live Terminal Interface for Real-Time Testing
├── demo.py                        # Standalone Scenario Runner & Verification Harness
├── benchmark_airbnb.py            # Official 50-Question Airbnb Compliance Benchmark
│
├── airbnb-grounding-rag-kb/       # Authoritative Airbnb Knowledge Base Corpus
│   ├── cleaned/                   # 20 Authoritative Markdown Policy Documents
│   └── evaluation/                # 50-Case Ground-Truth Benchmark Dataset
│
├── frontend/                      # Web Dashboard & Demonstration UI
│   ├── run.py                     # 1-Click Launch Script (auto-opens browser)
│   ├── server.py                  # FastAPI REST API Server
│   ├── test_frontend.py           # Automated Frontend Endpoint Test Suite
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

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more details.

---

<div align="center">
<b>ControlPlane.ai — Enterprise Responsible AI Gateway</b><br>
<i>Engineered for Maximum Safety, Verifiable Grounding, and Sub-15ms Latency SLAs.</i>
</div>
