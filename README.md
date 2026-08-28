# ControlPlane.ai — Responsible AI Control Plane (Round 2 Prototype)

> **Problem Statement 1 (PS1)**: A modular, decoupled Responsible AI Control Plane featuring pre-execution guardrails, fast parallel heuristics, RAG factual grounding verification, AI-as-a-Judge compliance evaluation, 3-tier policy arbitration, and cryptographic SHA-256 hash-chained audit logging.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Interactive Live Terminal (Enter Custom Prompts)
```bash
python cli.py
```

### 3. Run the Benchmark Demo (All 6 Scenarios)
```bash
python demo.py
```

### 3. Run with Live Groq API (Optional)
```bash
# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_groq_api_key_here"
python demo.py

# Linux / macOS
export GROQ_API_KEY="gsk_your_groq_api_key_here"
python demo.py
```

### 4. Run the Pytest Test Suite
```bash
pytest -v
```

---

## 📁 Repository Structure

```
AIC_codebase/
├── logging_config.py       # Centralized JSON logging & ContextVar Trace ID propagation
├── models.py               # Shared Pydantic data schemas, thresholds & enterprise knowledge base
├── llm_client.py           # Unified Groq API client with structured JSON mode
├── pii.py                  # Stage 1: Pre-Execution Guardrails (Luhn validation, reverse-offset slice redaction)
├── fast_checks.py          # Stage 3A: Fast Parallel Checks (Prioritized PII, banned lexicon & N-gram loop/entropy scorer)
├── rag_verifier.py         # Stage 3B: RAG Grounding Verification (RetEngine + Hybrid NLI entailment layer)
├── ai_judge.py             # Stage 3C: AI-as-a-Judge Sequential Evaluation (Bias, Tone, Policy compliance)
├── arbitrator.py           # Stage 4: Policy Arbitration (Composite risk score, FinCheck & 3-tier routing matrix)
├── audit_hitl.py           # Stage 5: Governance & Audit (O(1) reverse-seek SHA-256 hash chain & HITL review queue)
├── pipeline.py             # Master pipeline orchestrator with waterfall latency telemetry
├── demo.py                 # Interactive benchmark runner for the 6 core enterprise scenarios
├── requirements.txt        # Minimal Python dependencies
├── .gitignore              # Ignores runtime logs, caches, and secrets
├── SYSTEM_DOCUMENTATION.md # Complete architectural documentation and step-by-step execution guide
└── tests/                  # Unit test suite across all stages
    ├── test_pii.py
    ├── test_fast_checks.py
    ├── test_rag_verifier.py
    ├── test_arbitrator.py
    └── test_audit_hitl.py
```

---

## 📖 Comprehensive Documentation
For the full architectural breakdown, mathematical scoring formulas, function dictionaries, and compliance proofs, see **[SYSTEM_DOCUMENTATION.md](SYSTEM_DOCUMENTATION.md)**.
