"""
pipeline.py - Master Pipeline Orchestrator for ControlPlane.ai
ControlPlane.ai (PS1 Architecture)

Executes the complete 5-Stage Responsible AI Control Plane lifecycle:
1. Stage 1: Pre-Execution Guardrails (Fast PII & Prompt Injection Filter)
2. Primary LLM Generation: Invokes PrimLLM on sanitized prompt
3. Stage 3A: Fast Parallel Checks (Heuristics & Statistical Scorer via ParallelBus)
4. Stage 3B: RAG Grounding Verification (Enterprise Retriever & Factual Verifier)
5. Stage 3C: AI-as-a-Judge Sequential Evaluation (Bias, Tone, Policy)
6. Stage 4: Policy Arbitration & Risk Assessment (Composite Risk Score, FinCheck, TierCheck)
7. Stage 5: Immutable Audit Logging (SHA-256 Hash Chain) & HITL Review Queue
"""

import time
import logging
from typing import Dict, Any, Optional

from models import (
    PipelineOutput,
    DecisionTier,
    HITLAction,
    ExecutionMode,
    CRAGStatus,
    Config
)
from pii import filter_input_pii_and_injection
from llm_client import call_llm
from fast_checks import run_stage3a_fast_checks
from rag_verifier import (
    verify_factual_grounding,
    retrieve_knowledge_chunks_scored,
    evaluate_crag_retrieval_confidence
)
from ai_judge import run_ai_judge
from arbitrator import arbitrate_decision, route_output, check_financial_trigger
from audit_hitl import log_audit_entry, hitl_queue_manager
from logging_config import set_trace_id, clear_trace_id

logger = logging.getLogger(__name__)


def run_controlplane(
    prompt: str,
    user_id: str = "default_user",
    auto_hitl_action: Optional[HITLAction] = None,
    log_path: str = "audit_log.jsonl"
) -> PipelineOutput:
    """
    Executes the end-to-end ControlPlane.ai inspection pipeline with Adaptive Latency Routing.

    Args:
        prompt: Raw user input text.
        user_id: Identifier of the requesting client or user.
        auto_hitl_action: Optional simulated human action (APPROVE | EDIT | OVERRIDE).
        log_path: Path to the immutable SHA-256 audit log file.

    Returns:
        PipelineOutput with delivered response, decision tier, telemetry, and audit hash.
    """
    t_start = time.perf_counter()
    waterfall: Dict[str, float] = {}
    trace_id = set_trace_id()
    logger.info(f"Pipeline started for user '{user_id}' [Adaptive Mode]", extra={"stage": "init", "component": "pipeline"})

    # ========================================================================
    # STAGE 1: Pre-Execution Guardrails (Input PII & Injection Filter <1ms)
    # ========================================================================
    t0 = time.perf_counter()
    stage1_res = filter_input_pii_and_injection(prompt)
    waterfall["stage1_pre_guardrails_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Early Termination on Critical Jailbreak Attack
    if stage1_res.is_blocked:
        arbitration = arbitrate_decision(
            prompt=prompt,
            candidate_response="",
            stage1_res=stage1_res
        )
        total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
        telemetry = {
            "user_id": user_id,
            "routing_mode": "ADAPTIVE",
            "active_path": "FAST",
            "total_latency_ms": total_time_ms,
            "waterfall_latency_ms": waterfall,
            "stage1_pre_guardrails": stage1_res.model_dump(),
            "candidate_response": "",
            "early_block": True
        }
        audit_entry = log_audit_entry(prompt, arbitration, telemetry, log_path=log_path)
        final_text = route_output(arbitration, "")
        
        try:
            import db
            db.save_interaction(
                trace_id=trace_id,
                prompt=prompt,
                decision=arbitration.decision.value,
                composite_score=arbitration.composite_score,
                final_response=final_text,
                latency_ms=total_time_ms,
                audit_hash=audit_entry.entry_hash
            )
        except Exception:
            pass

        return PipelineOutput(
            final_response=final_text,
            decision=arbitration.decision,
            composite_score=arbitration.composite_score,
            is_financial_trigger=arbitration.is_financial_trigger,
            telemetry=telemetry,
            audit_hash=audit_entry.entry_hash,
            execution_mode=ExecutionMode.ADAPTIVE,
            active_path="FAST"
        )

    # ========================================================================
    # STAGE 2: Primary LLM Generation (BM25 Context Retrieval + CRAG Eval)
    # ========================================================================
    t0 = time.perf_counter()
    scored_chunks = retrieve_knowledge_chunks_scored(stage1_res.sanitized_prompt, top_k=3)
    kb_chunks = [c for _, c in scored_chunks]
    
    crag_conf, crag_status = evaluate_crag_retrieval_confidence(
        query_text=stage1_res.sanitized_prompt,
        retrieved_chunks=kb_chunks,
        scored_list=scored_chunks
    )

    if kb_chunks and crag_status != CRAGStatus.KNOWLEDGE_GAP_ABSTAIN:
        policy_context = "\n\n".join(f"[{c.title}]: {c.content}" for c in kb_chunks)
        system_instruction = (
            f"{Config.SYSTEM_PERSONA}\n\n"
            f"Use the following official policies to answer the user's inquiry accurately:\n\n"
            f"{policy_context}"
        )
    else:
        system_instruction = Config.SYSTEM_PERSONA

    try:
        candidate_response = call_llm(
            prompt=stage1_res.sanitized_prompt,
            system_instruction=system_instruction
        )
        if isinstance(candidate_response, dict):
            candidate_response = str(candidate_response)
    except Exception as e:
        # Fallback candidate if no API key is provided
        candidate_response = f"Processed response for request: '{stage1_res.sanitized_prompt}'."
    waterfall["primary_llm_generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ========================================================================
    # STAGE 3A: Fast Parallel Checks (Heuristics & Statistical Scorer)
    # ========================================================================
    t0 = time.perf_counter()
    stage3a_res = run_stage3a_fast_checks(candidate_response, prompt)
    waterfall["stage3a_fast_checks_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ========================================================================
    # ADAPTIVE LATENCY ROUTER: Fast-Path vs Deep-Path Elevation
    # ========================================================================
    is_fin_trigger, _ = check_financial_trigger(prompt, candidate_response)

    # Dynamically elevate to Deep-Path if triggers or ambiguity detected
    needs_deep = (
        is_fin_trigger or
        crag_status != CRAGStatus.HIGH_CONFIDENCE or
        stage3a_res.heuristic_risk > 2.0 or
        stage3a_res.stat_risk > 2.0
    )
    if needs_deep:
        active_path = "DEEP"
        use_nli = True
        run_judge = True
    else:
        active_path = "FAST"
        use_nli = False
        run_judge = False

    # ========================================================================
    # STAGE 3B: RAG Grounding Verification (RetEngine + RAGVerifier)
    # ========================================================================
    t0 = time.perf_counter()
    stage3b_res = verify_factual_grounding(
        candidate_response=candidate_response,
        query=prompt,
        use_nli=use_nli
    )
    waterfall["stage3b_rag_grounding_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ========================================================================
    # STAGE 3C: AI-as-a-Judge Evaluation (Executed on DEEP path)
    # ========================================================================
    t0 = time.perf_counter()
    if run_judge:
        stage3c_res = run_ai_judge(prompt, candidate_response, stage3a_res, stage3b_res)
    else:
        stage3c_res = None
    waterfall["stage3c_ai_judge_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ========================================================================
    # STAGE 4: Policy Arbitration & Risk Assessment (CalcScore, FinCheck, TierCheck)
    # ========================================================================
    t0 = time.perf_counter()
    arbitration = arbitrate_decision(
        prompt=prompt,
        candidate_response=candidate_response,
        stage1_res=stage1_res,
        stage3a_res=stage3a_res,
        stage3b_res=stage3b_res,
        stage3c_res=stage3c_res
    )
    waterfall["stage4_arbitration_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ========================================================================
    # STAGE 5: Governance, HITL Queue & Immutable Audit Logging
    # ========================================================================
    t0 = time.perf_counter()
    quarantined_ticket_id = None
    delivered_response = None

    if arbitration.decision == DecisionTier.HITL:
        # Enqueue quarantined ticket
        ticket = hitl_queue_manager.enqueue(prompt, candidate_response, arbitration)
        quarantined_ticket_id = ticket.ticket_id

        # Optional auto-resolve for inline testing
        if auto_hitl_action:
            resolved_ticket = hitl_queue_manager.resolve_ticket(
                ticket.ticket_id,
                action=auto_hitl_action,
                reviewer_notes="Inline simulated review."
            )
            delivered_response = resolved_ticket.final_delivered_text

    if delivered_response is None:
        delivered_response = route_output(arbitration, candidate_response, quarantined_ticket_id)

    waterfall["stage5_governance_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

    # Assemble comprehensive telemetry trace
    telemetry = {
        "user_id": user_id,
        "routing_mode": "ADAPTIVE",
        "active_path": active_path,
        "total_latency_ms": total_time_ms,
        "waterfall_latency_ms": waterfall,
        "stage1_pre_guardrails": stage1_res.model_dump(),
        "candidate_response": candidate_response,
        "stage3a_fast_checks": stage3a_res.model_dump(),
        "stage3b_rag_grounding": stage3b_res.model_dump(),
        "stage3c_ai_judge": stage3c_res.model_dump() if stage3c_res else {},
        "stage4_arbitration": arbitration.model_dump(),
        "quarantined_ticket_id": quarantined_ticket_id
    }

    # Write immutable SHA-256 hash-chained audit entry
    audit_entry = log_audit_entry(prompt, arbitration, telemetry, log_path=log_path)

    # Mirror interaction to SQLite database
    try:
        import db
        db.save_interaction(
            trace_id=trace_id,
            prompt=prompt,
            decision=arbitration.decision.value,
            composite_score=arbitration.composite_score,
            final_response=delivered_response,
            latency_ms=total_time_ms,
            audit_hash=audit_entry.entry_hash
        )
    except Exception:
        pass

    return PipelineOutput(
        final_response=delivered_response,
        decision=arbitration.decision,
        composite_score=arbitration.composite_score,
        is_financial_trigger=arbitration.is_financial_trigger,
        ticket_id=quarantined_ticket_id,
        telemetry=telemetry,
        audit_hash=audit_entry.entry_hash,
        execution_mode=ExecutionMode.ADAPTIVE,
        active_path=active_path
    )
