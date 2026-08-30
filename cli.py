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
from models import DecisionTier, HITLAction, ExecutionMode
import db


def print_divider():
    print("-" * 80)


def print_banner():
    print("\n" + "=" * 80)
    print("  CONTROLPLANE.AI - INTERACTIVE LIVE TERMINAL")
    print("  Type any customer or host prompt to inspect the 5-stage pipeline live.")
    print("  Commands:")
    print("    'status'         - View database counts (KB docs, pending tickets, reviews)")
    print("    'history'        - View recent conversation history across ALLOW, HITL, & BLOCK")
    print("    'kb'             - List loaded Airbnb policy documents")
    print("    'pending'        - View all quarantined HITL tickets")
    print("    'verify'         - Verify SHA-256 cryptographic audit chain continuity")
    print("    'exit'/'q'       - Quit the interactive session")
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

        cmd = prompt.lower()

        if cmd in ["exit", "quit", "q"]:
            print("Exiting interactive session. Goodbye!")
            break

        if cmd == "status":
            chunks = db.get_all_knowledge_chunks()
            pending = db.list_pending_hitl_tickets()
            metrics = db.get_policy_tuning_metrics_from_db()
            print("\nDatabase Status (controlplane.db):")
            print(f"  - Knowledge Base Documents: {len(chunks)}")
            print(f"  - Pending HITL Tickets:     {len(pending)}")
            print(f"  - Total Feedback Reviews:   {metrics['total_reviews']}")
            print(f"  - Approval / Allow Rate:    {metrics['allow_rate'] * 100:.1f}%\n")
            continue

        if cmd == "kb":
            chunks = db.get_all_knowledge_chunks()
            print(f"\nAuthoritative Policy Documents in Knowledge Base ({len(chunks)} total):")
            for idx, c in enumerate(chunks, 1):
                print(f"  {idx:02d}. [{c.doc_id}] ({c.category}/{c.product}/{c.region}) -> {c.title}")
            print()
            continue

        if cmd == "pending":
            pending = hitl_queue_manager.list_pending_tickets()
            print(f"\nPending Quarantined HITL Tickets ({len(pending)} total):")
            if not pending:
                print("  (No pending tickets in review queue)\n")
            else:
                for t in pending:
                    print(f"  - [{t.ticket_id}] Reason: {t.reason} | Score: {t.composite_score:.2f}")
                    print(f"    Prompt: \"{t.prompt}\"\n")
            continue

        if cmd == "history":
            interactions = db.list_interactions(limit=15)
            print(f"\nRecent Conversation History in SQLite ({len(interactions)} total):")
            if not interactions:
                print("  (No conversations recorded in SQLite yet)\n")
            else:
                for idx, item in enumerate(interactions, 1):
                    dec_str = f"[{item['decision']}]"
                    print(f"  {idx:02d}. {dec_str:7s} | Score: {item['composite_score']:4.2f} | Latency: {item['latency_ms']:6.1f}ms | Time: {item['timestamp'][:19]}")
                    print(f"      Q: \"{item['prompt'][:60]}...\"")
                    print(f"      A: \"{item['final_response'][:80]}...\"\n")
            continue

        if cmd == "verify":
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
            print(f"  - PII Masked: {[p['entity_type'] for p in s1['pii_detected']]}")
            print(f"  - Sanitized Prompt: \"{s1.get('sanitized_prompt')}\"")
        else:
            print("  - PII Status: Clean (No PII detected)")
            
        if s1.get("is_injection"):
            print(f"  - Injection Alert: {s1.get('block_reason')} (Score: {s1.get('injection_score')}/10)")
        else:
            print("  - Injection Status: Clean (No adversarial patterns)")

        # If early block triggered
        if telemetry.get("early_block"):
            print("\n[EARLY TERMINATION TRIGGERED]")
            print(f"  - Decision: {format_decision(output.decision)}")
            print(f"  - Final Output: \"{output.final_response}\"")
            print(f"  - SHA-256 Audit Hash: {output.audit_hash[:16]}...")
            continue

        # 2. Stage 2: Primary LLM Generation
        candidate = telemetry.get("candidate_response", "")
        print(f"\n[STAGE 2: Primary LLM Generation] ({telemetry.get('waterfall_latency_ms', {}).get('primary_llm_generation_ms', 0):.2f} ms)")
        preview = (candidate[:180] + "...") if len(candidate) > 180 else candidate
        print(f"  - Candidate Draft: \"{preview}\"")

        # 3. Stage 3A: Fast Parallel Checks
        s3a = telemetry.get("stage3a_fast_checks", {})
        print(f"\n[STAGE 3A: Fast Parallel Checks] ({telemetry.get('waterfall_latency_ms', {}).get('stage3a_fast_checks_ms', 0):.2f} ms)")
        print(f"  - Heuristic Risk: {s3a.get('heuristic_risk', 0):.2f}/10 | Banned Hits: {s3a.get('banned_lexicon_hits', [])}")
        print(f"  - Stat Repetition Loop: {s3a.get('ngram_repetition', 0):.2f} | Entropy: {s3a.get('perplexity_score', 0):.2f} bits | Stat Risk: {s3a.get('stat_risk', 0):.2f}/10")

        # 4. Stage 3B: RAG Grounding Verification
        s3b = telemetry.get("stage3b_rag_grounding", {})
        print(f"\n[STAGE 3B: RAG Grounding Verification] ({telemetry.get('waterfall_latency_ms', {}).get('stage3b_rag_grounding_ms', 0):.2f} ms)")
        docs = [d.get("doc_id") for d in s3b.get("retrieved_chunks", [])]
        print(f"  - Retrieved Policy Chunks: {docs or 'None (General Inquiry)'}")
        print(f"  - CRAG Status: {s3b.get('crag_status', 'N/A')} | Retrieval Confidence (ρ): {s3b.get('crag_confidence', 1.0):.2f}")
        print(f"  - Grounding Score: {s3b.get('grounding_score', 10):.2f}/10 (RAG Risk: {s3b.get('rag_risk', 0):.2f}/10)")
        print(f"  - Verification Status: {s3b.get('verification_status', 'N/A')} | Confidence: {s3b.get('verification_confidence', 1.0):.2f}")
        if s3b.get("numeric_mismatches"):
            print(f"  - Numeric Mismatches: {s3b.get('numeric_mismatches')}")
        if s3b.get("unsupported_claims"):
            print(f"  - Unsupported Claims: {len(s3b.get('unsupported_claims'))} found")

        # 5. Stage 3C: AI-as-a-Judge
        s3c = telemetry.get("stage3c_ai_judge", {})
        print(f"\n[STAGE 3C: AI-as-a-Judge Evaluation] ({telemetry.get('waterfall_latency_ms', {}).get('stage3c_ai_judge_ms', 0):.2f} ms)")
        print(f"  - Bias Score: {s3c.get('bias_score', 0):.2f}/10 | Tone Score: {s3c.get('tone_score', 0):.2f}/10 | Policy Risk: {s3c.get('policy_risk_score', 0):.2f}/10")
        print(f"  - Judge Notes: \"{s3c.get('judge_notes', '')}\"")

        # 6. Stage 4: Policy Arbitration
        s4 = telemetry.get("stage4_arbitration", {})
        print(f"\n[STAGE 4: Policy Arbitration] ({telemetry.get('waterfall_latency_ms', {}).get('stage4_arbitration_ms', 0):.2f} ms)")
        print(f"  - Composite Risk Score: \033[1m{output.composite_score:.2f} / 10.0\033[0m")
        print(f"  - Financial Trigger: {output.is_financial_trigger}")
        print(f"  - Final Decision: {format_decision(output.decision)}")
        print(f"  - Arbitration Reason: {s4.get('reason')}")

        # 7. Stage 5: Delivered Output & Governance
        print_divider()
        print(f"\033[1mDELIVERED RESPONSE TO USER:\033[0m")
        print(output.final_response)
        print_divider()
        print(f"- Total Latency: {telemetry.get('total_latency_ms', 0):.2f} ms | Cryptographic SHA-256 Hash: {output.audit_hash[:16]}...")

        # Interactive HITL Review if Quarantined
        if output.decision == DecisionTier.HITL and output.ticket_id:
            print(f"\n\033[93m[HITL ACTION REQUIRED]\033[0m Ticket '{output.ticket_id}' is pending in the Human Review Queue.")
            while True:
                choice = input("Select Human Action [allow / edit / block]: ").strip().lower()
                
                if choice in ("allow", "approve"):
                    res = hitl_queue_manager.resolve_ticket(output.ticket_id, HITLAction.ALLOW, reviewer_notes="Allowed via CLI.")
                    print(f"\n\033[92m[TICKET RESOLVED: ALLOW]\033[0m Released Original Candidate:\n\"{res.final_delivered_text}\"")
                    break
                elif choice == "edit":
                    edited = input("Enter corrected response to deliver: ").strip()
                    res = hitl_queue_manager.resolve_ticket(output.ticket_id, HITLAction.EDIT, edited_text=edited, reviewer_notes="Edited via CLI.")
                    print(f"\n\033[94m[TICKET RESOLVED: EDIT]\033[0m Delivered Human-Edited Response:\n\"{res.final_delivered_text}\"")
                    break
                elif choice in ("block", "override"):
                    res = hitl_queue_manager.resolve_ticket(output.ticket_id, HITLAction.BLOCK, reviewer_notes="Blocked via CLI.")
                    print(f"\n\033[91m[TICKET RESOLVED: BLOCK]\033[0m Response Permanently Blocked. Final Status: {res.status}")
                    break
                else:
                    print("Invalid selection. Please choose one of: 'allow', 'edit', or 'block'.")


if __name__ == "__main__":
    run_interactive_session()
