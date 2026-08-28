"""
cli.py - Interactive Live Terminal Interface for ControlPlane.ai
ControlPlane.ai (PS1 Architecture)

Allows you to enter custom user prompts and inspect the complete 5-stage pipeline
execution, waterfall latencies, risk score breakdowns, and decision routing live.
"""

import os
import sys
from pipeline import run_controlplane
from audit_hitl import hitl_queue_manager, verify_audit_log_integrity
from models import DecisionTier, HITLAction


def print_divider():
    print("-" * 80)


def print_banner():
    print("\n" + "=" * 80)
    print("  CONTROLPLANE.AI — INTERACTIVE LIVE TERMINAL")
    print("  Type any prompt to inspect the 5-stage Control Plane in real-time.")
    print("  Commands: 'exit' or 'q' to quit | 'verify' to check SHA-256 audit log")
    print("=" * 80 + "\n")


def format_decision(decision: DecisionTier) -> str:
    if decision == DecisionTier.ALLOW:
        return f"\033[92m[ALLOW - Stream to User]\033[0m"
    elif decision == DecisionTier.HITL:
        return f"\033[93m[HITL - Quarantined for Human Review]\033[0m"
    elif decision == DecisionTier.BLOCK:
        return f"\033[91m[BLOCK - Safe Canned Fallback]\033[0m"
    return decision.value


def run_interactive_session():
    print_banner()

    while True:
        try:
            prompt = input("\n\033[1mEnter Prompt >>> \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive session. Goodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() in ["exit", "quit", "q"]:
            print("Exiting interactive session. Goodbye!")
            break

        if prompt.lower() == "verify":
            valid, msg = verify_audit_log_integrity("audit_log.jsonl")
            print(f"\nAudit Log Status: {'[PASSED]' if valid else '[FAILED]'}")
            print(f"Proof: {msg}\n")
            continue

        print("\n" + "." * 80)
        print("  Running ControlPlane.ai 5-Stage Inspection Pipeline...")
        print("." * 80)

        # Run pipeline
        output = run_controlplane(prompt, log_path="audit_log.jsonl")
        telemetry = output.telemetry

        # 1. Stage 1: Pre-Execution Guardrails
        s1 = telemetry.get("stage1_pre_guardrails", {})
        print(f"\n[STAGE 1: Pre-Execution Guardrails] ({telemetry.get('waterfall_latency_ms', {}).get('stage1_pre_guardrails_ms', 0):.2f} ms)")
        if s1.get("pii_detected"):
            print(f"  • PII Masked: {[p['entity_type'] for p in s1['pii_detected']]}")
            print(f"  • Sanitized Prompt: \"{s1.get('sanitized_prompt')}\"")
        else:
            print("  • PII Status: Clean (No PII detected)")
            
        if s1.get("is_injection"):
            print(f"  • Injection Alert: {s1.get('block_reason')} (Score: {s1.get('injection_score')}/10)")
        else:
            print("  • Injection Status: Clean (No adversarial patterns)")

        # If early block triggered
        if telemetry.get("early_block"):
            print("\n[EARLY TERMINATION TRIGGERED]")
            print(f"  • Decision: {format_decision(output.decision)}")
            print(f"  • Final Output: \"{output.final_response}\"")
            print(f"  • SHA-256 Audit Hash: {output.audit_hash[:16]}...")
            continue

        # 2. Stage 2: Primary LLM Generation
        candidate = telemetry.get("candidate_response", "")
        print(f"\n[STAGE 2: Primary LLM Generation] ({telemetry.get('waterfall_latency_ms', {}).get('primary_llm_generation_ms', 0):.2f} ms)")
        preview = (candidate[:180] + "...") if len(candidate) > 180 else candidate
        print(f"  • Candidate Draft: \"{preview}\"")

        # 3. Stage 3A: Fast Parallel Checks
        s3a = telemetry.get("stage3a_fast_checks", {})
        print(f"\n[STAGE 3A: Fast Parallel Checks] ({telemetry.get('waterfall_latency_ms', {}).get('stage3a_fast_checks_ms', 0):.2f} ms)")
        print(f"  • Heuristic Risk: {s3a.get('heuristic_risk', 0):.2f}/10 | Banned Hits: {s3a.get('banned_lexicon_hits', [])}")
        print(f"  • Stat Repetition Loop: {s3a.get('ngram_repetition', 0):.2f} | Entropy: {s3a.get('perplexity_score', 0):.2f} bits | Stat Risk: {s3a.get('stat_risk', 0):.2f}/10")

        # 4. Stage 3B: RAG Grounding Verification
        s3b = telemetry.get("stage3b_rag_grounding", {})
        print(f"\n[STAGE 3B: RAG Grounding Verification] ({telemetry.get('waterfall_latency_ms', {}).get('stage3b_rag_grounding_ms', 0):.2f} ms)")
        docs = [d.get("doc_id") for d in s3b.get("retrieved_chunks", [])]
        print(f"  • Retrieved Policy Chunks: {docs or 'None (General Inquiry)'}")
        print(f"  • Grounding Score: {s3b.get('grounding_score', 10):.2f}/10 (RAG Risk: {s3b.get('rag_risk', 0):.2f}/10)")
        if s3b.get("numeric_mismatches"):
            print(f"  • Numeric Mismatches: {s3b.get('numeric_mismatches')}")
        if s3b.get("unsupported_claims"):
            print(f"  • Unsupported Claims: {len(s3b.get('unsupported_claims'))} found")

        # 5. Stage 3C: AI-as-a-Judge
        s3c = telemetry.get("stage3c_ai_judge", {})
        print(f"\n[STAGE 3C: AI-as-a-Judge Evaluation] ({telemetry.get('waterfall_latency_ms', {}).get('stage3c_ai_judge_ms', 0):.2f} ms)")
        print(f"  • Bias Score: {s3c.get('bias_score', 0):.2f}/10 | Tone Score: {s3c.get('tone_score', 0):.2f}/10 | Policy Risk: {s3c.get('policy_risk_score', 0):.2f}/10")
        print(f"  • Judge Notes: \"{s3c.get('judge_notes', '')}\"")

        # 6. Stage 4: Policy Arbitration
        s4 = telemetry.get("stage4_arbitration", {})
        print(f"\n[STAGE 4: Policy Arbitration] ({telemetry.get('waterfall_latency_ms', {}).get('stage4_arbitration_ms', 0):.2f} ms)")
        print(f"  • Composite Risk Score: \033[1m{output.composite_score:.2f} / 10.0\033[0m")
        print(f"  • Financial Trigger: {output.is_financial_trigger}")
        print(f"  • Final Decision: {format_decision(output.decision)}")
        print(f"  • Arbitration Reason: {s4.get('reason')}")

        # 7. Stage 5: Delivered Output & Governance
        print_divider()
        print(f"\033[1mDELIVERED RESPONSE TO USER:\033[0m")
        print(output.final_response)
        print_divider()
        print(f"• Total Latency: {telemetry.get('total_latency_ms', 0):.2f} ms | Cryptographic SHA-256 Hash: {output.audit_hash[:16]}...")

        # Interactive HITL Review if Quarantined
        if output.decision == DecisionTier.HITL and output.ticket_id:
            print(f"\n\033[93m[HITL ACTION REQUIRED]\033[0m Ticket '{output.ticket_id}' is pending in the Human Review Queue.")
            choice = input("Resolve ticket now? [approve / edit / override / skip]: ").strip().lower()
            
            if choice == "approve":
                res = hitl_queue_manager.resolve_ticket(output.ticket_id, HITLAction.APPROVE, reviewer_notes="Approved via CLI.")
                print(f"✓ Approved. Released Original Candidate:\n\"{res.final_delivered_text}\"")
            elif choice == "edit":
                edited = input("Enter corrected response to deliver: ").strip()
                res = hitl_queue_manager.resolve_ticket(output.ticket_id, HITLAction.EDIT, edited_text=edited, reviewer_notes="Edited via CLI.")
                print(f"✓ Delivered Edited Response:\n\"{res.final_delivered_text}\"")
            elif choice == "override":
                res = hitl_queue_manager.resolve_ticket(output.ticket_id, HITLAction.OVERRIDE, reviewer_notes="Blocked via CLI.")
                print(f"✓ Force Blocked. Final Status: {res.status}")


if __name__ == "__main__":
    run_interactive_session()
