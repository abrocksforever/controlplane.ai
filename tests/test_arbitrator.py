"""
test_arbitrator.py - Tests for Stage 4: Policy Arbitration & Risk Assessment
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from arbitrator import (
    check_financial_trigger,
    calculate_composite_score,
    arbitrate_decision,
    route_output,
)
from models import (
    DecisionTier,
    Stage1Result,
    Stage3AResult,
    Stage3BResult,
    Stage3CResult,
    ArbitrationResult,
    Config,
)


# ============================================================================
# Financial Trigger Detection Tests
# ============================================================================

class TestFinancialTrigger:
    """Tests for FinCheck financial trigger detection."""

    def test_wire_transfer_triggers(self):
        is_trigger, reason = check_financial_trigger(
            "Process a wire transfer of $50,000", ""
        )
        assert is_trigger is True
        assert "wire transfer" in reason.lower()

    def test_large_dollar_triggers(self):
        is_trigger, _ = check_financial_trigger(
            "", "The amount of $5000 has been disbursed"
        )
        assert is_trigger is True

    def test_loan_approval_triggers(self):
        is_trigger, _ = check_financial_trigger(
            "Request loan approval for the applicant", ""
        )
        assert is_trigger is True

    def test_routing_number_triggers(self):
        is_trigger, _ = check_financial_trigger(
            "Routing number 021000021 for the transfer", ""
        )
        assert is_trigger is True

    def test_safe_query_no_trigger(self):
        is_trigger, reason = check_financial_trigger(
            "What is the return policy?", "Items can be returned within 30 days."
        )
        assert is_trigger is False
        assert reason is None

    def test_small_dollar_no_trigger(self):
        """Amounts < $1,000 should NOT trigger financial escalation."""
        is_trigger, _ = check_financial_trigger(
            "", "Refund of $50 has been processed"
        )
        assert is_trigger is False


# ============================================================================
# Composite Score Calculation Tests
# ============================================================================

class TestCompositeScore:
    """Tests for CalcScore weighted risk score computation."""

    def test_all_zero_risks_returns_zero(self):
        score, breakdown = calculate_composite_score(
            stage3a_res=Stage3AResult(heuristic_risk=0.0, stat_risk=0.0),
            stage3b_res=Stage3BResult(rag_risk=0.0),
            stage3c_res=Stage3CResult(judge_risk_score=0.0)
        )
        assert score == 0.0

    def test_weights_sum_correctly(self):
        """Verify weighted combination produces expected result."""
        score, breakdown = calculate_composite_score(
            stage3a_res=Stage3AResult(heuristic_risk=10.0, stat_risk=10.0),
            stage3b_res=Stage3BResult(rag_risk=10.0),
            stage3c_res=Stage3CResult(judge_risk_score=10.0)
        )
        # All dimensions at 10.0 -> composite = sum of all weights * 10 = 10.0
        assert score == 10.0

    def test_catastrophic_single_dimension_elevates_score(self):
        """If any single dimension >= 9.0, composite should be elevated."""
        score, _ = calculate_composite_score(
            stage3a_res=Stage3AResult(heuristic_risk=9.5, stat_risk=0.0),
            stage3b_res=Stage3BResult(rag_risk=0.0),
            stage3c_res=Stage3CResult(judge_risk_score=0.0)
        )
        assert score >= 9.0  # Should be elevated from weighted 2.375 to >= 9.0

    def test_stage1_blocked_returns_max_score(self):
        """Critical injection block should override to 10.0."""
        blocked_result = Stage1Result(
            sanitized_prompt="blocked",
            is_blocked=True,
            injection_score=9.5
        )
        score, _ = calculate_composite_score(stage1_res=blocked_result)
        assert score == 10.0

    def test_score_capped_at_10(self):
        score, _ = calculate_composite_score(
            stage3a_res=Stage3AResult(heuristic_risk=10.0, stat_risk=10.0),
            stage3b_res=Stage3BResult(rag_risk=10.0),
            stage3c_res=Stage3CResult(judge_risk_score=10.0)
        )
        assert score <= 10.0

    def test_score_never_negative(self):
        score, _ = calculate_composite_score()
        assert score >= 0.0


# ============================================================================
# Decision Tier Tests
# ============================================================================

class TestArbitrateDecision:
    """Tests for TierCheck 3-tier decision matrix."""

    def test_low_risk_allows(self):
        result = arbitrate_decision(
            prompt="Hello",
            candidate_response="Hi there!",
            stage3a_res=Stage3AResult(heuristic_risk=0.0, stat_risk=0.0),
            stage3b_res=Stage3BResult(rag_risk=0.0),
            stage3c_res=Stage3CResult(judge_risk_score=0.0)
        )
        assert result.decision == DecisionTier.ALLOW
        assert result.composite_score <= Config.ALLOW_THRESHOLD

    def test_high_risk_blocks(self):
        result = arbitrate_decision(
            prompt="",
            candidate_response="",
            stage3a_res=Stage3AResult(heuristic_risk=9.5, stat_risk=5.0),
            stage3b_res=Stage3BResult(rag_risk=8.0),
            stage3c_res=Stage3CResult(judge_risk_score=8.0)
        )
        assert result.decision == DecisionTier.BLOCK
        assert result.composite_score >= Config.BLOCK_THRESHOLD

    def test_financial_trigger_forces_hitl(self):
        result = arbitrate_decision(
            prompt="Wire transfer $50,000 to account 12345",
            candidate_response="Processing wire transfer.",
            stage3a_res=Stage3AResult(heuristic_risk=0.0, stat_risk=0.0),
            stage3b_res=Stage3BResult(rag_risk=0.0),
            stage3c_res=Stage3CResult(judge_risk_score=0.0)
        )
        assert result.decision == DecisionTier.HITL
        assert result.is_financial_trigger is True

    def test_mid_range_score_routes_to_hitl(self):
        result = arbitrate_decision(
            prompt="Query",
            candidate_response="Response",
            stage3a_res=Stage3AResult(heuristic_risk=5.0, stat_risk=3.0),
            stage3b_res=Stage3BResult(rag_risk=5.0),
            stage3c_res=Stage3CResult(judge_risk_score=4.0)
        )
        assert result.decision == DecisionTier.HITL
        assert result.composite_score > Config.ALLOW_THRESHOLD
        assert result.composite_score < Config.BLOCK_THRESHOLD

    def test_stage1_blocked_forces_block(self):
        blocked = Stage1Result(
            sanitized_prompt="blocked",
            is_blocked=True,
            block_reason="DAN jailbreak",
            injection_score=9.5
        )
        result = arbitrate_decision(
            prompt="malicious",
            candidate_response="",
            stage1_res=blocked
        )
        assert result.decision == DecisionTier.BLOCK
        assert result.fallback_response is not None


# ============================================================================
# Output Router Tests
# ============================================================================

class TestRouteOutput:
    """Tests for OutputRouter delivery logic."""

    def test_allow_returns_candidate(self):
        arb = ArbitrationResult(
            composite_score=1.0,
            decision=DecisionTier.ALLOW,
            is_financial_trigger=False,
        )
        output = route_output(arb, "Original response")
        assert output == "Original response"

    def test_block_returns_fallback(self):
        arb = ArbitrationResult(
            composite_score=9.0,
            decision=DecisionTier.BLOCK,
            is_financial_trigger=False,
            fallback_response="Safe fallback"
        )
        output = route_output(arb, "Dangerous response")
        assert output == "Safe fallback"

    def test_hitl_returns_queue_message(self):
        arb = ArbitrationResult(
            composite_score=5.0,
            decision=DecisionTier.HITL,
            is_financial_trigger=True,
            reason="Financial trigger"
        )
        output = route_output(arb, "Response", "TICKET-001")
        assert "human review" in output.lower() or "human agent" in output.lower()
        assert "TICKET-001" in output
