"""
benchmark_airbnb.py - 50-Case Grounding & Compliance Benchmark Harness
ControlPlane.ai (PS1 Architecture)

Runs the official 50-question Airbnb Grounding Benchmark Suite:
- Validates Factual Accuracy (18 ALLOW cases)
- Validates Hallucination & Contradiction Interception (25 BLOCK cases)
- Validates Ambiguity Quarantine (7 FLAG cases)
- Measures Stage 1-5 Waterfall Latencies
- Mathematically verifies 100% continuous SHA-256 hash-chain audit log integrity

Usage:
  python benchmark_airbnb.py             # Deterministic Offline Mode (Fast, CI/CD friendly)
  python benchmark_airbnb.py --live      # Live Mode (Calls Primary LLM API)
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any

from pipeline import run_controlplane
from models import DecisionTier, ArbitrationResult, Stage1Result, ExecutionMode
from fast_checks import run_stage3a_fast_checks
from rag_verifier import verify_factual_grounding
from ai_judge import run_ai_judge
from arbitrator import arbitrate_decision, route_output
from audit_hitl import log_audit_entry, verify_audit_log_integrity, hitl_queue_manager
import logging_config

DATASET_PATH = "airbnb-grounding-rag-kb/evaluation/benchmark_dataset.json"
BENCHMARK_LOG_PATH = "benchmark_audit.jsonl"


def load_benchmark_dataset() -> List[Dict[str, Any]]:
    """Loads the 50 ground-truth evaluation cases."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Benchmark dataset not found at '{DATASET_PATH}'.")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_offline_case(case: Dict[str, Any], log_path: str) -> Dict[str, Any]:
    """Runs a single benchmark case in deterministic offline mode."""
    t0_total = time.perf_counter()
    question = case["question"]
    candidate = case["canonical_candidate_response"]
    expected = case["expected_verdict"]

    trace_id = logging_config.set_trace_id()
    waterfall = {}

    # Stage 1: Pre-Guardrails
    t0 = time.perf_counter()
    from pii import filter_input_pii_and_injection
    s1_res = filter_input_pii_and_injection(question)
    waterfall["stage1_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Stage 3A: Fast Parallel Checks
    t0 = time.perf_counter()
    s3a_res = run_stage3a_fast_checks(candidate, question)
    waterfall["stage3a_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Stage 3B: RAG Grounding Verification (Deterministic without NLI for offline)
    t0 = time.perf_counter()
    s3b_res = verify_factual_grounding(candidate, query=question, use_nli=False)
    waterfall["stage3b_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Stage 3C: AI Judge (Deterministic simulated evaluation)
    t0 = time.perf_counter()
    from models import Stage3CResult
    s3c_res = Stage3CResult(
        bias_score=0.0,
        tone_score=0.0,
        policy_risk_score=0.0 if s3b_res.grounding_score >= 7.0 else (10.0 - s3b_res.grounding_score),
        judge_risk_score=0.0 if s3b_res.grounding_score >= 7.0 else (10.0 - s3b_res.grounding_score),
        judge_notes="Deterministic benchmark evaluation."
    )
    waterfall["stage3c_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Stage 4: Arbitration
    t0 = time.perf_counter()
    arb_res = arbitrate_decision(
        prompt=question,
        candidate_response=candidate,
        stage1_res=s1_res,
        stage3a_res=s3a_res,
        stage3b_res=s3b_res,
        stage3c_res=s3c_res
    )
    waterfall["stage4_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Stage 5: Logging
    t0 = time.perf_counter()
    total_latency_ms = round((time.perf_counter() - t0_total) * 1000, 2)
    telemetry = {
        "user_id": "benchmark_user",
        "routing_mode": "ADAPTIVE",
        "active_path": "FAST" if s3b_res.grounding_score >= 7.0 else "DEEP",
        "total_latency_ms": total_latency_ms,
        "waterfall_latency_ms": waterfall,
        "stage1_pre_guardrails": s1_res.model_dump(),
        "candidate_response": candidate,
        "stage3a_fast_checks": s3a_res.model_dump(),
        "stage3b_rag_grounding": s3b_res.model_dump(),
        "stage3c_ai_judge": s3c_res.model_dump() if s3c_res else {},
        "stage4_arbitration": arb_res.model_dump()
    }
    audit_entry = log_audit_entry(question, arb_res, telemetry, log_path=log_path)
    waterfall["stage5_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    actual = arb_res.decision.value
    is_match = (
        (expected == "ALLOW" and actual == "ALLOW") or
        (expected == "BLOCK" and actual in ("BLOCK", "HITL")) or
        (expected == "FLAG" and actual == "HITL")
    )

    return {
        "id": case["id"],
        "question": question,
        "expected": expected,
        "actual": actual,
        "composite_score": arb_res.composite_score,
        "grounding_score": s3b_res.grounding_score,
        "verification_status": s3b_res.verification_status.value,
        "crag_status": s3b_res.crag_status.value,
        "crag_confidence": s3b_res.crag_confidence,
        "is_match": is_match,
        "total_latency_ms": total_latency_ms,
        "audit_hash": audit_entry.entry_hash
    }


def run_benchmark(live_mode: bool = False):
    """Executes the complete 50-case benchmark harness."""
    print("\n" + "=" * 80)
    print(f"  CONTROLPLANE.AI — AIRBNB GROUNDING & COMPLIANCE BENCHMARK (50 CASES)")
    print(f"  Routing: [ADAPTIVE] | Live Primary LLM: {'ENABLED' if live_mode else 'OFFLINE (Deterministic)'}")
    print("=" * 80 + "\n")

    cases = load_benchmark_dataset()

    if os.path.exists(BENCHMARK_LOG_PATH):
        os.remove(BENCHMARK_LOG_PATH)

    results = []
    t_start = time.perf_counter()

    for idx, case in enumerate(cases, 1):
        if live_mode:
            t0 = time.perf_counter()
            out = run_controlplane(case["question"], log_path=BENCHMARK_LOG_PATH)
            lat = round((time.perf_counter() - t0) * 1000, 2)
            actual = out.decision.value
            expected = case["expected_verdict"]
            is_match = (
                (expected == "ALLOW" and actual == "ALLOW") or
                (expected == "BLOCK" and actual in ("BLOCK", "HITL")) or
                (expected == "FLAG" and actual == "HITL")
            )
            res = {
                "id": case["id"],
                "question": case["question"],
                "expected": expected,
                "actual": actual,
                "composite_score": out.composite_score,
                "grounding_score": out.telemetry.get("stage3b_rag_grounding", {}).get("grounding_score", 10.0),
                "verification_status": out.telemetry.get("stage3b_rag_grounding", {}).get("verification_status", "UNKNOWN"),
                "crag_status": out.telemetry.get("stage3b_rag_grounding", {}).get("crag_status", "UNKNOWN"),
                "is_match": is_match,
                "total_latency_ms": lat,
                "audit_hash": out.audit_hash
            }
        else:
            res = run_single_offline_case(case, BENCHMARK_LOG_PATH)

        results.append(res)
        status_sym = "[PASS]" if res["is_match"] else "[FAIL]"
        print(f"[{res['id']:02d}/50] {status_sym:6s} | Expected: {res['expected']:5s} -> Actual: {res['actual']:5s} | Score: {res['composite_score']:4.2f} | Latency: {res['total_latency_ms']:6.2f}ms | Q: \"{res['question'][:45]}...\"")

    total_time = round(time.perf_counter() - t_start, 2)

    # Compute Statistics & Metrics
    total = len(results)
    passed = sum(1 for r in results if r["is_match"])
    allow_cases = [r for r in results if r["expected"] == "ALLOW"]
    block_cases = [r for r in results if r["expected"] == "BLOCK"]
    flag_cases = [r for r in results if r["expected"] == "FLAG"]

    allow_correct = sum(1 for r in allow_cases if r["actual"] == "ALLOW")
    block_intercepted = sum(1 for r in block_cases if r["actual"] in ("BLOCK", "HITL"))
    flag_quarantined = sum(1 for r in flag_cases if r["actual"] == "HITL")

    latencies = [r["total_latency_ms"] for r in results]
    latencies.sort()
    p50_lat = latencies[len(latencies) // 2]
    p90_lat = latencies[int(len(latencies) * 0.9)]
    avg_lat = sum(latencies) / len(latencies)

    print("\n" + "=" * 80)
    print("  GROUNDING & COMPLIANCE EVALUATION SCORECARD")
    print("=" * 80)
    print(f"- Total Evaluated Cases:         {total}")
    print(f"- Overall Safety Compliance:     {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"- Autonomous Deflection (ALLOW): {allow_correct}/{len(allow_cases)} ({allow_correct/max(1,len(allow_cases))*100:.1f}%)")
    print(f"- Hallucination Intercept (BLOCK):{block_intercepted}/{len(block_cases)} ({block_intercepted/max(1,len(block_cases))*100:.1f}%)")
    print(f"- Ambiguity Quarantine (FLAG):    {flag_quarantined}/{len(flag_cases)} ({flag_quarantined/max(1,len(flag_cases))*100:.1f}%)")
    print(f"- Latency Profile:               Avg: {avg_lat:.2f}ms | P50: {p50_lat:.2f}ms | P90: {p90_lat:.2f}ms")
    print(f"- Total Benchmark Wall Time:     {total_time:.2f}s")
    print("-" * 80)

    # Cryptographic Audit Log Verification
    is_valid, proof_msg = verify_audit_log_integrity(BENCHMARK_LOG_PATH)
    print(f"Cryptographic Audit Log Verification ({BENCHMARK_LOG_PATH}):")
    print(f"  Status: {'[PASSED - 100% Continuous SHA-256 Hash Chain]' if is_valid else '[FAILED]'}")
    print(f"  Proof:  {proof_msg}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Airbnb Grounding Benchmark Runner")
    parser.add_argument("--live", action="store_true", help="Run against live Primary LLM API")
    args = parser.parse_args()
    run_benchmark(live_mode=args.live)
