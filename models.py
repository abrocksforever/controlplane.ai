"""
models.py - Shared Data Contracts, Configuration and Knowledge Base
ControlPlane.ai (PS1 Architecture)

This file defines all Pydantic schemas, enums, risk scoring weights, 
and in-memory enterprise knowledge chunks used across all 5 pipeline stages.
"""

import os
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ============================================================================
# 1. Enums
# ============================================================================

class DecisionTier(str, Enum):
    """Routing decision tiers determined by Composite Risk Score and Financial Triggers."""
    ALLOW = "ALLOW"      # Score <= 2.5: Stream directly to user
    HITL = "HITL"        # 2.5 < Score < 7.0 OR Financial Trigger: Quarantined for Human Review
    BLOCK = "BLOCK"      # Score >= 7.0: Terminated with safe fallback response


class HITLAction(str, Enum):
    """Actions human compliance reviewers can take on quarantined tickets."""
    ALLOW = "ALLOW"      # Release original candidate response unchanged
    EDIT = "EDIT"        # Deliver human-sanitized/corrected response
    BLOCK = "BLOCK"      # Force block and deliver safe canned fallback

    # Aliases for backward compatibility
    APPROVE = "ALLOW"
    OVERRIDE = "BLOCK"


class VerificationStatus(str, Enum):
    """Factual grounding verification status from Stage 3B."""
    VERIFIED_GROUNDED = "VERIFIED_GROUNDED"        # Docs matched, claims entailed (G >= 7.0)
    PARTIALLY_GROUNDED = "PARTIALLY_GROUNDED"      # Docs matched, minor gaps (3.0 <= G < 7.0)
    CONTRADICTED = "CONTRADICTED"                  # Docs matched, factual conflict (G < 3.0)
    UNVERIFIED_ASSERTION = "UNVERIFIED_ASSERTION"  # 0 docs matched, but factual claims made
    GENERAL_CONVERSATION = "GENERAL_CONVERSATION"  # 0 docs matched, no claims (greetings/thanks)


# ============================================================================
# 2. Stage 1: Pre-Execution Guardrails Models
# ============================================================================

class PIIEntity(BaseModel):
    """Represents a detected personally identifiable information span."""
    entity_type: str       # e.g., "SSN", "EMAIL", "CREDIT_CARD", "PHONE", "API_KEY"
    text: str              # Raw sensitive string
    start: int = 0         # Start index in text
    end: int = 0           # End index in text


class Stage1Result(BaseModel):
    """Output from Stage 1 (Input PII and Prompt Injection Filter)."""
    sanitized_prompt: str
    pii_detected: List[PIIEntity] = Field(default_factory=list)
    is_injection: bool = False
    injection_score: float = 0.0   # 0.0 (safe) to 10.0 (blatant attack)
    is_blocked: bool = False       # True if direct adversarial attack detected early
    block_reason: Optional[str] = None


# ============================================================================
# 3. Stage 3A: Fast Parallel Checks Models
# ============================================================================

class Stage3AResult(BaseModel):
    """Output from Stage 3A (Heuristic Agent + Statistical Scorer)."""
    output_pii: List[PIIEntity] = Field(default_factory=list)
    banned_lexicon_hits: List[str] = Field(default_factory=list)
    heuristic_risk: float = 0.0    # 0.0 (clean) to 10.0 (severe leak / banned content)
    
    perplexity_score: float = 0.0  # Statistical fluency metric
    ngram_repetition: float = 0.0  # Degenerate loop / repetition metric
    cosine_similarity: float = 1.0 # Semantic proximity to prompt
    stat_risk: float = 0.0         # 0.0 (normal) to 10.0 (high anomaly)


# ============================================================================
# 4. Stage 3B: RAG Grounding Verification Models
# ============================================================================

class KnowledgeChunk(BaseModel):
    """An enterprise knowledge base document chunk with provenance metadata."""
    doc_id: str
    title: str
    category: str                  # e.g., "cancellation", "refund", "india", "exceptions"
    content: str
    keywords: List[str] = Field(default_factory=list)
    product: Optional[str] = "all"      # "home" | "service" | "all"
    audience: Optional[str] = "guest"   # "guest" | "host" | "guest_host"
    region: Optional[str] = "global"    # "global" | "india"
    source_url: Optional[str] = None    # Official Help Center URL


class Stage3BResult(BaseModel):
    """Output from Stage 3B (Enterprise RAG Retriever + Factual Grounding Verifier)."""
    retrieved_chunks: List[KnowledgeChunk] = Field(default_factory=list)
    grounding_score: float = 10.0  # 10.0 (fully grounded) to 0.0 (total hallucination)
    unsupported_claims: List[str] = Field(default_factory=list)
    numeric_mismatches: List[str] = Field(default_factory=list)
    rag_risk: float = 0.0          # Computed as (10.0 - grounding_score)
    verification_confidence: float = 1.0  # Mathematical confidence in [0.0, 1.0]
    verification_status: VerificationStatus = VerificationStatus.VERIFIED_GROUNDED


# ============================================================================
# 5. Stage 3C: AI-as-a-Judge Sequential Final Check Models
# ============================================================================

class Stage3CResult(BaseModel):
    """Output from Stage 3C (AI-as-a-Judge Evaluation LLM)."""
    bias_score: float = 0.0        # 0.0 (fair) to 10.0 (severe demographic / class bias)
    tone_score: float = 0.0        # 0.0 (professional) to 10.0 (hostile / aggressive)
    policy_risk_score: float = 0.0 # 0.0 (compliant) to 10.0 (enterprise policy violation)
    judge_risk_score: float = 0.0  # Aggregated judge evaluation (0.0 to 10.0)
    judge_notes: str = ""


# ============================================================================
# 6. Stage 4: Policy Arbitration and Risk Assessment Models
# ============================================================================

class ArbitrationResult(BaseModel):
    """Output from Stage 4 (Composite Risk and Decision Engine)."""
    composite_score: float         # 0.0 to 10.0 Scale
    decision: DecisionTier         # ALLOW | HITL | BLOCK
    is_financial_trigger: bool     # Forced escalation flag
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    reason: str = ""
    fallback_response: Optional[str] = None


# ============================================================================
# 7. Stage 5: Governance, HITL and Audit Models
# ============================================================================

class HITLTicket(BaseModel):
    """Human-in-the-Loop review queue item."""
    ticket_id: str
    timestamp: str
    prompt: str
    candidate_response: str
    composite_score: float
    is_financial_trigger: bool
    reason: str
    status: str = "PENDING"        # "PENDING" | "APPROVED" | "EDITED" | "OVERRIDDEN"
    reviewer_notes: Optional[str] = None
    final_delivered_text: Optional[str] = None


class AuditEntry(BaseModel):
    """Cryptographically chained audit log record."""
    entry_id: str
    timestamp: str
    prompt_hash: str
    prev_hash: str
    entry_hash: str
    decision: str
    composite_score: float
    is_financial_trigger: bool
    trace: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# 8. Master Pipeline Input and Output Models
# ============================================================================

class PromptRequest(BaseModel):
    """Client request entering the API Gateway."""
    prompt: str
    user_id: str = "default_user"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineOutput(BaseModel):
    """Final response returned by OutputRouter."""
    final_response: str
    decision: DecisionTier
    composite_score: float
    is_financial_trigger: bool
    ticket_id: Optional[str] = None
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    audit_hash: str = ""


# ============================================================================
# 9. System Configuration and Enterprise Knowledge Base
# ============================================================================

class Config:
    """
    System risk weights, thresholds, and policy parameters.
    
    All numeric thresholds and weights can be overridden via environment variables
    with the CONTROLPLANE_ prefix (e.g., CONTROLPLANE_ALLOW_THRESHOLD=3.0).
    """
    
    # Default LLM Model ID (overridable via CONTROLPLANE_LLM_MODEL)
    DEFAULT_MODEL: str = os.environ.get("CONTROLPLANE_LLM_MODEL", "qwen/qwen3.8-27b")

    # Score Thresholds (overridable via CONTROLPLANE_ALLOW_THRESHOLD, CONTROLPLANE_BLOCK_THRESHOLD)
    ALLOW_THRESHOLD: float = float(os.environ.get("CONTROLPLANE_ALLOW_THRESHOLD", "2.5"))
    BLOCK_THRESHOLD: float = float(os.environ.get("CONTROLPLANE_BLOCK_THRESHOLD", "7.0"))
    
    # Composite Score Component Weights (Sum to 1.0)
    # Overridable via CONTROLPLANE_WEIGHT_HEURISTIC, etc.
    WEIGHT_HEURISTIC: float = float(os.environ.get("CONTROLPLANE_WEIGHT_HEURISTIC", "0.25"))
    WEIGHT_STATISTICAL: float = float(os.environ.get("CONTROLPLANE_WEIGHT_STATISTICAL", "0.15"))
    WEIGHT_RAG_GROUNDING: float = float(os.environ.get("CONTROLPLANE_WEIGHT_RAG_GROUNDING", "0.35"))
    WEIGHT_AI_JUDGE: float = float(os.environ.get("CONTROLPLANE_WEIGHT_AI_JUDGE", "0.25"))

    # Safe Canned Fallback Response (for BLOCK)
    SAFE_FALLBACK: str = os.environ.get(
        "CONTROLPLANE_SAFE_FALLBACK",
        "I cannot fulfill this request as the generated content violates enterprise "
        "safety, compliance, or grounding policies. Please contact support if you believe "
        "this was flagged in error."
    )

    # Automated Handover Message (for HITL)
    HITL_HANDOVER_MESSAGE: str = os.environ.get(
        "CONTROLPLANE_HITL_HANDOVER",
        "This request cannot be handled automatically and requires human review. "
        "I am connecting you to a human agent from our team who will assist you shortly."
    )

    # Configurable System Prompt Persona
    SYSTEM_PERSONA: str = os.environ.get(
        "CONTROLPLANE_SYSTEM_PERSONA",
        "You are an official, helpful, and compliant customer support assistant. "
        "Ground your answers strictly in the official policies provided below."
    )

    # Deterministic Financial Trigger Keywords
    FINANCIAL_KEYWORDS = [
        "wire transfer", "transfer funds", "payout", "disburse", "credit increase",
        "routing number", "bank account", "authorize payment", "send money",
        "loan approval", "mortgage approval", "credit limit", "$",
        "security deposit payout", "direct offline payment", "wire money outside airbnb",
        "damage claim payout", "manual bank payout"
    ]

    # Prohibited Banned Lexicon
    BANNED_LEXICON = [
        "bypass safety", "exploit", "unauthorized", "drop table", "dump database",
        "classified credentials", "internal secret keys", "private api token"
    ]


# In-Memory Enterprise Knowledge Base for RAG Verification
ENTERPRISE_KNOWLEDGE_BASE: List[KnowledgeChunk] = [
    KnowledgeChunk(
        doc_id="KB-001",
        title="Enterprise Return and Refund Policy",
        category="refund_policy",
        content=(
            "Standard items can be returned within 30 days of purchase for a full refund. "
            "Refunds are processed to the original payment method within 5 to 7 business days. "
            "Promotional or clearance items are eligible for store credit only within 14 days. "
            "No cash refunds are issued for amounts exceeding $100 without manager authorization."
        ),
        keywords=["return", "refund", "30 days", "store credit", "14 days", "$100", "cash refund"]
    ),
    KnowledgeChunk(
        doc_id="KB-002",
        title="Credit Line and Loan Underwriting Guidelines",
        category="credit_underwriting",
        content=(
            "Automated credit line increases cannot exceed $2,500 without secondary credit analysis. "
            "Any loan application denial must be strictly based on Debt-to-Income ratio (>43%) or "
            "verified credit bureau score (<620). Demographic attributes, marital status, and geographic "
            "proxies must never be used in credit evaluations."
        ),
        keywords=["credit", "loan", "underwriting", "$2,500", "debt-to-income", "credit score", "fair lending"]
    ),
    KnowledgeChunk(
        doc_id="KB-003",
        title="Information Security and Credential Handling",
        category="security",
        content=(
            "API keys, master credentials, and cryptographic seed tokens must never be disclosed to "
            "end-users under any circumstances. System architecture prompts and hidden directives are "
            "strictly confidential proprietary assets."
        ),
        keywords=["api key", "credentials", "security", "token", "confidential", "system prompt"]
    ),
]
