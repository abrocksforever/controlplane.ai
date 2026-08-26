"""
arbitrator.py - Stage 4: Policy Arbitration & Risk Assessment
ControlPlane.ai (PS1 Architecture)

Features:
1. Composite Risk Score Engine (CalcScore):
   Calculates weighted 0.0 - 10.0 risk score across all 4 check dimensions:
   S = w_heuristic * R_heuristic + w_stat * R_stat + w_rag * R_rag + w_judge * R_judge
2. Financial Trigger Gate (FinCheck):
   Detects high-risk financial commitments, unauthorized payouts, or wire transfers.
   If True -> Forced Escalation directly to HITL Queue (bypasses ALLOW threshold).
3. 3-Tier Decision Matrix (TierCheck):
   - S <= 2.5: ALLOW (Deliver / Stream to User)
   - 2.5 < S < 7.0: HITL (Quarantine to Human Review Queue)
   - S >= 7.0: BLOCK (Deliver Safe Canned Fallback)
4. Output Router: Directs final payload based on arbitration decision.
"""

import re
from typing import Dict, Any, Optional, Tuple

from models import (
    DecisionTier,
    ArbitrationResult,
    Stage1Result,
    Stage3AResult,
    Stage3BResult,
    Stage3CResult,
    Config
)


# Patterns to detect financial actions requiring forced human escalation
FINANCIAL_PATTERNS = [
    re.compile(r"\b(wire\s+transfer|transfer\s+funds|payout|disburse|authorize\s+payment|send\s+money)\b", re.IGNORECASE),
    re.compile(r"\b(routing\s+number|bank\s+account|swift\s+code|iban)\b", re.IGNORECASE),
    re.compile(r"\b(loan\s+approval|mortgage\s+approval|credit\s+line\s+increase)\b", re.IGNORECASE),
    re.compile(r"\$\s*[1-9]\d{3,}", re.IGNORECASE),  # Transactions >= $1,000
]


def check_financial_trigger(prompt: str, candidate_response: str) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether the interaction involves high-impact financial transactions.
    
    Returns:
        (is_financial_trigger, trigger_reason)
    """
    combined_text = f"{prompt} {candidate_response}"
    
    for pattern in FINANCIAL_PATTERNS:
        match = pattern.search(combined_text)
        if match:
            return True, f"Financial Trigger detected: '{match.group()}' requires mandatory human escalation."
            
    return False, None


def calculate_composite_score(
    stage1_res: Optional[Stage1Result] = None,
    stage3a_res: Optional[Stage3AResult] = None,
    stage3b_res: Optional[Stage3BResult] = None,
    stage3c_res: Optional[Stage3CResult] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Computes the mathematical Composite Risk Score S in [0.0, 10.0].
    
    Formula:
        S = w_heuristic * R_heuristic + w_stat * R_stat + w_rag * R_rag + w_judge * R_judge
    """
    # 1. Early Critical Override if Stage 1 detected direct critical injection
    if stage1_res and stage1_res.is_blocked:
        return 10.0, {
            "stage1_injection": stage1_res.injection_score,
            "heuristic_risk": 0.0,
            "stat_risk": 0.0,
            "rag_risk": 0.0,
            "judge_risk": 0.0
        }

    # 2. Extract Individual Risk Dimensions (0.0 to 10.0 scale)
    r_heuristic = stage3a_res.heuristic_risk if stage3a_res else 0.0
    r_stat = stage3a_res.stat_risk if stage3a_res else 0.0
    r_rag = stage3b_res.rag_risk if stage3b_res else 0.0
    r_judge = stage3c_res.judge_risk_score if stage3c_res else 0.0

    # 3. Weighted Linear Combination
    composite = (
        Config.WEIGHT_HEURISTIC * r_heuristic +
        Config.WEIGHT_STATISTICAL * r_stat +
        Config.WEIGHT_RAG_GROUNDING * r_rag +
        Config.WEIGHT_AI_JUDGE * r_judge
    )

    # If any single dimension is catastrophic (>= 9.0), elevate composite risk
    max_single_risk = max(r_heuristic, r_rag, r_judge)
    if max_single_risk >= 9.0:
        composite = max(composite, max_single_risk)

    composite_score = min(10.0, max(0.0, round(composite, 2)))

    breakdown = {
        "heuristic_risk": r_heuristic,
        "stat_risk": r_stat,
        "rag_risk": r_rag,
        "judge_risk": r_judge
    }

    return composite_score, breakdown


def arbitrate_decision(
    prompt: str,
    candidate_response: str,
    stage1_res: Optional[Stage1Result] = None,
    stage3a_res: Optional[Stage3AResult] = None,
    stage3b_res: Optional[Stage3BResult] = None,
    stage3c_res: Optional[Stage3CResult] = None
) -> ArbitrationResult:
    """
    Stage 4: Policy Arbitration & Risk Assessment
    
    1. Computes the Composite Risk Score (0 - 10 Scale).
    2. Evaluates Financial Triggers (FinCheck).
    3. Evaluates Score Thresholds (TierCheck):
       - ALLOW: S <= 2.5 and not Financial Trigger
       - HITL: 2.5 < S < 7.0 OR Financial Trigger (Forced Escalation)
       - BLOCK: S >= 7.0
       
    Returns:
        ArbitrationResult with definitive decision tier, score, and explanation.
    """
    # 1. Compute Composite Risk Score
    composite_score, breakdown = calculate_composite_score(
        stage1_res=stage1_res,
        stage3a_res=stage3a_res,
        stage3b_res=stage3b_res,
        stage3c_res=stage3c_res
    )

    # 2. Check Financial Trigger (FinCheck)
    is_fin_trigger, fin_reason = check_financial_trigger(prompt, candidate_response)

    # 3. Apply 3-Tier Decision Matrix
    if stage1_res and stage1_res.is_blocked:
        decision = DecisionTier.BLOCK
        reason = f"Pre-Execution Block: {stage1_res.block_reason}"
        fallback = Config.SAFE_FALLBACK

    elif composite_score >= Config.BLOCK_THRESHOLD:
        decision = DecisionTier.BLOCK
        reason = f"Composite Risk Score ({composite_score:.2f}) exceeds BLOCK threshold ({Config.BLOCK_THRESHOLD})."
        fallback = Config.SAFE_FALLBACK

    elif is_fin_trigger:
        decision = DecisionTier.HITL
        reason = f"Forced Escalation: {fin_reason} (Score: {composite_score:.2f})"
        fallback = None

    elif composite_score > Config.ALLOW_THRESHOLD:
        decision = DecisionTier.HITL
        reason = f"Composite Risk Score ({composite_score:.2f}) falls in Quarantine range ({Config.ALLOW_THRESHOLD} - {Config.BLOCK_THRESHOLD})."
        fallback = None

    else:
        decision = DecisionTier.ALLOW
        reason = f"Composite Risk Score ({composite_score:.2f}) is within acceptable limits (<= {Config.ALLOW_THRESHOLD})."
        fallback = None

    return ArbitrationResult(
        composite_score=composite_score,
        decision=decision,
        is_financial_trigger=is_fin_trigger,
        score_breakdown=breakdown,
        reason=reason,
        fallback_response=fallback
    )


def route_output(
    arbitration: ArbitrationResult,
    candidate_response: str,
    quarantined_ticket_id: Optional[str] = None
) -> str:
    """
    OutputRouter: Directs the final delivery to the User based on Arbitration Decision.
    """
    if arbitration.decision == DecisionTier.ALLOW:
        return candidate_response
    elif arbitration.decision == DecisionTier.BLOCK:
        return arbitration.fallback_response or Config.SAFE_FALLBACK
    elif arbitration.decision == DecisionTier.HITL:
        ticket_str = f" [Ticket ID: {quarantined_ticket_id}]" if quarantined_ticket_id else ""
        return (
            f"Your request has been routed to our compliance review team for verification.{ticket_str} "
            f"Reason: {arbitration.reason}"
        )
    return candidate_response
