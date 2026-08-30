"""
test_tiered_crag.py - Unit Tests for CRAG & Tiered Latency Routing
ControlPlane.ai (PS1 Architecture)
"""

import pytest
import time
from models import (
    ExecutionMode,
    CRAGStatus,
    VerificationStatus,
    DecisionTier,
    Config
)
from rag_verifier import (
    evaluate_crag_retrieval_confidence,
    retrieve_knowledge_chunks,
    retrieve_knowledge_chunks_scored,
    verify_factual_grounding,
    _is_number_in_negation_clause
)
from pipeline import run_controlplane
from arbitrator import calculate_composite_score, arbitrate_decision


# ============================================================================
# 1. CRAG Retrieval Quality & Confidence Tests
# ============================================================================

def test_crag_high_confidence_on_exact_policy_query():
    """Exact policy query should yield high retrieval confidence (rho >= 0.70)."""
    query = "What is the refund timeline for UPI payments in India?"
    scored = retrieve_knowledge_chunks_scored(query, top_k=3)
    chunks = [c for _, c in scored]
    rho, status = evaluate_crag_retrieval_confidence(query, chunks, scored)
    
    assert len(chunks) > 0
    assert chunks[0].doc_id == "india_refunds"
    assert rho >= 0.70
    assert status == CRAGStatus.HIGH_CONFIDENCE


def test_crag_low_confidence_abstains_on_out_of_domain():
    """Unindexed out-of-domain query should yield rho < 0.40 and KNOWLEDGE_GAP_ABSTAIN."""
    query = "What is the cancellation refund policy for Tokyo bullet train tickets and Shinkansen rail passes?"
    scored = retrieve_knowledge_chunks_scored(query, top_k=3)
    chunks = [c for _, c in scored]
    rho, status = evaluate_crag_retrieval_confidence(query, chunks, scored)
    
    assert rho < 0.40
    assert status == CRAGStatus.KNOWLEDGE_GAP_ABSTAIN


def test_crag_active_abstention_gate_on_unindexed_factual_claim():
    """When CRAG confidence is low and empirical claims are asserted, verifier must actively abstain."""
    out_of_domain_claim = "Shinkansen bullet train refunds take 14 business days with a $50 processing fee."
    res = verify_factual_grounding(
        candidate_response=out_of_domain_claim,
        query="Can I get a Shinkansen bullet train refund?",
        use_nli=False
    )
    
    assert res.crag_status == CRAGStatus.KNOWLEDGE_GAP_ABSTAIN
    assert res.verification_status == VerificationStatus.UNVERIFIED_ASSERTION
    assert res.grounding_score == 2.50
    assert res.rag_risk == 7.50
    assert res.verification_confidence == 0.0
    assert any("CRAG Abstention" in claim for claim in res.unsupported_claims)


# ============================================================================
# 2. Negation-Aware Entity Grounding Tests (False Positive Elimination)
# ============================================================================

def test_negation_filter_identifies_denial_clauses():
    """Verifies that negation phrases preceding numbers are detected."""
    t1 = "Cancelling a non-refundable stay after 45 days does not qualify for a $2,000 cash refund."
    t2 = "You cannot receive a $500 payout outside the policy."
    t3 = "The guest won't be refunded the 100% fee."
    
    assert _is_number_in_negation_clause("$2,000", t1) is True
    assert _is_number_in_negation_clause("45 days", t1) is True
    assert _is_number_in_negation_clause("$500", t2) is True
    assert _is_number_in_negation_clause("100%", t3) is True


def test_negation_filter_denials_not_penalized_in_grounding():
    """A response that quotes invalid numbers to explicitly deny them must NOT receive numeric mismatch penalties."""
    refusal_response = (
        "Based on official policy, cancelling a non-refundable stay after 45 days "
        "does not qualify for a $2,000 cash refund. If you cancel, you won't be refunded."
    )
    res = verify_factual_grounding(
        candidate_response=refusal_response,
        query="Can I cancel a non-refundable stay after 45 days for a $2,000 refund?",
        use_nli=False
    )
    
    # Neither $2,000 nor 45 days should appear in numeric_mismatches because both are inside denial clauses
    assert len(res.numeric_mismatches) == 0
    assert res.grounding_score >= 7.0
    assert res.verification_status == VerificationStatus.VERIFIED_GROUNDED


def test_negation_filter_affirmative_hallucinations_still_penalized():
    """An affirmative claim making ungrounded numeric promises MUST be penalized."""
    affirmative_response = "You are guaranteed a $2,000 cash refund within 45 days of cancellation."
    res = verify_factual_grounding(
        candidate_response=affirmative_response,
        query="What refund do I get?",
        use_nli=False
    )
    
    # Affirmative numbers not in source policy MUST be flagged
    assert len(res.numeric_mismatches) >= 1
    assert any("$2,000" in m or "45 days" in m for m in res.numeric_mismatches)
    assert res.grounding_score < 7.0


# ============================================================================
# 3. Dynamic Weight Renormalization & Fast-Path Tests
# ============================================================================

def test_fast_path_weight_renormalization():
    """Fast-Path (stage3c_res is None) must renormalize weights over {0.25, 0.15, 0.35}."""
    from models import Stage3AResult, Stage3BResult
    s3a = Stage3AResult(heuristic_risk=2.0, stat_risk=1.0)
    s3b = Stage3BResult(grounding_score=8.0, rag_risk=2.0)
    
    score, breakdown = calculate_composite_score(
        stage3a_res=s3a,
        stage3b_res=s3b,
        stage3c_res=None
    )
    
    assert score == pytest.approx(1.80, abs=0.05)


def test_adaptive_fast_path_routine_query():
    """Routine high-confidence query runs on Fast-Path (<25ms, skips AI Judge)."""
    out = run_controlplane(prompt="What is the refund timeline for UPI payments in India?")
    
    assert out.active_path == "FAST"
    assert out.decision == DecisionTier.ALLOW
    assert out.telemetry["stage3c_ai_judge"] == {}
    assert "stage3c_ai_judge_ms" in out.telemetry["waterfall_latency_ms"]


def test_adaptive_elevates_on_financial_trigger():
    """Adaptive router automatically elevates to DEEP path on high-liability financial actions."""
    out = run_controlplane(prompt="Authorize a wire transfer payout of $5,000 to external host account 12345.")
    
    assert out.active_path == "DEEP"
    assert out.is_financial_trigger is True
    assert out.decision in (DecisionTier.HITL, DecisionTier.BLOCK)
