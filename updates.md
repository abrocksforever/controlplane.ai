# ControlPlane.ai — Architectural Updates & Improvement Roadmap

This document outlines completed fixes, current component implementations, and planned enhancements across each module in the **ControlPlane.ai** pipeline (PS1 Architecture).

---

## 1. Summary of Completed Components & Fixes

### A. Data Contracts & Configuration (`models.py`)
- **Implemented**: Complete Pydantic schemas for all 5 stages (`Stage1Result`, `Stage3AResult`, `Stage3BResult`, `Stage3CResult`, `ArbitrationResult`, `HITLTicket`, `AuditEntry`, `PipelineOutput`).
- **Configuration**: Standardized score thresholds ($S \le 2.5 \to \text{ALLOW}, S \ge 7.0 \to \text{BLOCK}$), composite weights, financial trigger keywords, and built-in enterprise knowledge base chunks.

### B. Unified LLM Interface (`llm_client.py`)
- **Implemented**: Single entry point `call_llm(prompt, system_instruction, json_mode, model)`.
- **Groq Integration**: Native support for Groq API (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`) with `response_format={"type": "json_object"}` for structured AI Judge outputs.
- **Removed**: Removed all hardcoded mock keywords and mock simulation branches in favor of clean, direct API execution.

### C. Stage 1: Pre-Execution Guardrails (`pii.py`)
- **Completed Critical Fixes**:
  1. *String Offset Corruption Fix*: Swapped naive `.replace()` with **Reverse-Offset Slicing** (`_redact_by_reverse_offset`), sorting entities by start index in descending order so earlier offsets are never corrupted.
  2. *Interval Overlap Collision Fix*: Implemented `_has_interval_overlap()` checking character intervals `[start, end)` rather than exact string equality.
  3. *Severity-Weighted Injection Scoring*: Replaced raw hit counts with weighted signatures (Critical attacks like DAN/Overrides $\to 9.0-9.5$, Mild extraction probes $\to 5.5-7.0$).
  4. *ReDoS-Safe Bounded Patterns*: Bounded regex lookups with proper word/digit boundary definitions.

### D. Stage 3A: Fast Parallel Checks (`fast_checks.py`)
- **Implemented**:
  1. *Heuristic Agent*: Scans candidate output in prioritized entity order (`API_KEY`, `CREDIT_CARD`, `SSN`, `EMAIL`, `PHONE`) and detects enterprise banned lexicon terms.
  2. *Statistical Scorer*: Computes N-Gram repetition ratio ($N=3$ for loop detection), Shannon Entropy (perplexity proxy), and morphological prefix content overlap.
  3. *Parallel Execution Bus*: Concurrent thread pool scatter-gather execution ($<20\text{ms}$ latency SLA).

---

## 2. Active Updates & Enhancements by Stage

### E. Stage 3B: RAG Grounding Verification (`rag_verifier.py`)

#### Current Evaluation Analysis:
| Aspect | Status | Analysis |
| :--- | :--- | :--- |
| **Numeric Strictness** | **Strong** | Regex entity extraction accurately catches fabricated numbers (`$500` vs `$100`, `90 days` vs `30 days`). |
| **General Chat Fallback** | **Good** | If no knowledge chunks match (e.g. conversational greetings), defaults to `grounding_score = 10.0` to avoid false alarms. |
| **Token-Overlap Claim Check** | **Needs Upgrade** | Raw token overlap causes *False Positives* on valid rephrasings/synonyms and *False Negatives* on inverted negations (e.g., "We do NOT offer refunds"). |

#### Planned Fix (Hybrid NLI Verification):
1. **Deterministic Fast Path (Keep)**: Exact number and currency regex matching for hard policy boundaries.
2. **Semantic NLI Entailment Layer (Upgrade)**:
   - Use `call_llm(json_mode=True)` with an NLI prompt to verify whether candidate claims are logically *entailed*, *contradicted*, or *unsupported* by the retrieved source document text.
   - Accurately captures negations, polarity flips, and semantic paraphrases.

---

### F. Stage 3C: AI-as-a-Judge Sequential Evaluation (`ai_judge.py`)
- **Status**: Ready for implementation.
- **Key Capabilities**:
  - `JudgeGate`: Aggregates the candidate response + all prior check evidence (PII findings, banned words, statistical loop metrics, and RAG grounding scores).
  - `AIJudge`: Calls Groq with a specialized compliance persona to evaluate 3 orthogonal dimensions:
    1. `bias_score` (Demographic disparity, protected class proxy bias).
    2. `tone_score` (Hostile, aggressive, unprofessional tone).
    3. `policy_risk_score` (Direct company policy breaches).
  - Returns validated JSON structure with explanatory notes.

---

### G. Stage 4: Policy Arbitration & Risk Assessment (`arbitrator.py`)
- **Status**: Pending.
- **Key Capabilities**:
  - **Composite Risk Score Calculation ($S \in [0.0, 10.0]$)**:
    $$S = w_{\text{heuristic}} \cdot R_{\text{heuristic}} + w_{\text{stat}} \cdot R_{\text{stat}} + w_{\text{rag}} \cdot (10 - G_{\text{RAG}}) + w_{\text{judge}} \cdot R_{\text{judge}}$$
  - **Financial Trigger Gate (`FinCheck`)**:
    - Scans for unauthorized transactions, payouts, wire transfers, or credit increases.
    - If triggered $\to$ **Forced Escalation directly to HITL Queue**, regardless of numerical score.
  - **3-Tier Routing Logic**:
    - $S \le 2.5 \to$ `ALLOW` (Stream to user)
    - $2.5 < S < 7.0$ or Financial Trigger $\to$ `HITL` (Quarantine for human review)
    - $S \ge 7.0 \to$ `BLOCK` (Safe canned fallback response)

---

### H. Stage 5: Governance, Audit & Continuous Learning (`audit_hitl.py`)
- **Status**: Pending.
- **Key Capabilities**:
  - **Immutable SHA-256 Hash Chained Audit Log**:
    - Every inspection trace is hashed: $H_i = \text{SHA256}(H_{i-1} + \text{Payload}_i)$.
    - Appended to `audit_log.jsonl`.
  - **Cryptographic Audit Integrity Verifier**:
    - `verify_audit_log_integrity()` function iterates over the chain and verifies tamper-evident proof.
  - **HITL Queue Management & Active Feedback**:
    - Supports review actions: `APPROVE`, `EDIT`, `OVERRIDE`.
    - Captures reviewer annotations to tune sensitivity thresholds over time.

---

### I. Master Orchestrator & CLI Runner (`pipeline.py` & `demo.py`)
- **`pipeline.py`**: Clean, modular orchestrator linking Stages 1 $\to$ Primary LLM $\to$ 3A $\to$ 3B $\to$ 3C $\to$ 4 $\to$ 5.
- **`demo.py`**: Executes the 6 core benchmark scenarios:
  1. *Safe Return Policy Query* $\to$ `ALLOW`
  2. *Financial Wire Transfer Trigger* $\to$ `HITL` (Forced Escalation)
  3. *Input PII Redaction* $\to$ `[REDACTED_SSN]` passed safely
  4. *Ungrounded / Fabricated Claim* $\to$ `HITL`
  5. *Adversarial Prompt Injection* $\to$ `BLOCK`
  6. *Audit Log SHA-256 Verification & HITL Resolution* $\to$ Cryptographic proof of integrity

---

## 3. Decoupled File Map

```
c:\Users\abroc\Desktop\AIC\AIC_codebase\
│
├── models.py         # [Done] Shared Pydantic data schemas, risk weights & knowledge base
├── llm_client.py     # [Done] Groq API caller with retry/timeout for generation & JSON evaluation
├── pii.py            # [Done] Stage 1: Input PII redaction (Luhn-validated) & weighted injection guard
├── fast_checks.py    # [Done] Stage 3A: Output PII, banned lexicon & statistical scorer
├── rag_verifier.py   # [Done] Stage 3B: RAG retriever & hybrid NLI grounding verifier
├── ai_judge.py       # [Done] Stage 3C: AI-as-a-Judge sequential evaluation
├── arbitrator.py     # [Done] Stage 4: Composite risk, FinCheck & decision router
├── audit_hitl.py     # [Done] Stage 5: SHA-256 Audit logger (thread-safe) & HITL queue manager
├── pipeline.py       # [Done] Master pipeline orchestrator
├── demo.py           # [Done] End-to-end scenario validation runner
├── logging_config.py # [Done] Structured JSON logging with trace IDs
├── tests/            # [Done] pytest test suite (PII, RAG, arbitrator, audit)
└── updates.md        # [Done] Architecture changelog and roadmap document
```
