"""
audit_hitl.py - Stage 5: Governance, Audit & Continuous Learning
ControlPlane.ai (PS1 Architecture)

Features:
1. Immutable Audit Log (AuditLog):
   Appends structured inspection traces with cryptographic SHA-256 Hash Chaining:
   H_i = SHA256(H_{i-1} + Payload_i).
2. Tamper-Evidence Verifier:
   Mathematically verifies the integrity of the audit log chain.
3. Human-In-The-Loop Triage Queue (HITLQueue):
   Manages quarantined interactions with Approve, Edit, and Override actions.
4. Active Feedback Store & Policy Matrix Optimizer:
   Captures human decisions and computes threshold calibration updates.
"""

import os
import json
import hashlib
import datetime
from typing import List, Dict, Any, Optional, Tuple

from models import (
    AuditEntry,
    HITLTicket,
    HITLAction,
    ArbitrationResult
)


DEFAULT_LOG_PATH = "audit_log.jsonl"
GENESIS_HASH = "0" * 64


# ============================================================================
# 1. Immutable SHA-256 Hash Chained Audit Logger
# ============================================================================

def _calculate_sha256(data: str) -> str:
    """Calculates SHA-256 hexadecimal hash string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def get_latest_audit_hash(log_path: str = DEFAULT_LOG_PATH) -> str:
    """Retrieves the hash of the most recent audit entry in the chain."""
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return GENESIS_HASH

    latest_hash = GENESIS_HASH
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    latest_hash = record.get("entry_hash", latest_hash)
    except Exception:
        return GENESIS_HASH

    return latest_hash


def log_audit_entry(
    prompt: str,
    arbitration: ArbitrationResult,
    telemetry_trace: Dict[str, Any],
    log_path: str = DEFAULT_LOG_PATH
) -> AuditEntry:
    """
    Appends an immutable, SHA-256 hash-chained entry to audit_log.jsonl.
    
    Hash chaining guarantee:
        entry_hash = SHA256(prev_hash + prompt_hash + decision + score + trace_json)
    """
    prev_hash = get_latest_audit_hash(log_path)
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    prompt_hash = _calculate_sha256(prompt)
    entry_id = f"AUDIT-{int(datetime.datetime.now().timestamp() * 1000)}"

    # Construct canonical payload for hashing
    payload_dict = {
        "entry_id": entry_id,
        "timestamp": now_iso,
        "prompt_hash": prompt_hash,
        "prev_hash": prev_hash,
        "decision": arbitration.decision.value,
        "composite_score": arbitration.composite_score,
        "is_financial_trigger": arbitration.is_financial_trigger,
        "trace": telemetry_trace
    }

    # Deterministic JSON string for hashing
    canonical_payload_str = json.dumps(payload_dict, sort_keys=True)
    entry_hash = _calculate_sha256(f"{prev_hash}:{canonical_payload_str}")

    audit_entry = AuditEntry(
        entry_id=entry_id,
        timestamp=now_iso,
        prompt_hash=prompt_hash,
        prev_hash=prev_hash,
        entry_hash=entry_hash,
        decision=arbitration.decision.value,
        composite_score=arbitration.composite_score,
        is_financial_trigger=arbitration.is_financial_trigger,
        trace=telemetry_trace
    )

    # Append to JSONL log
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(audit_entry.model_dump_json() + "\n")

    return audit_entry


def verify_audit_log_integrity(log_path: str = DEFAULT_LOG_PATH) -> Tuple[bool, str]:
    """
    Cryptographically verifies the entire SHA-256 Hash Chain in audit_log.jsonl.
    
    Returns:
        (is_valid: bool, verification_message: str)
    """
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return True, "Audit log is empty (Genesis state)."

    expected_prev_hash = GENESIS_HASH
    line_number = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line_number += 1
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception as e:
                return False, f"Tamper / Corruption detected at line {line_number}: Invalid JSON ({e})"

            # 1. Verify prev_hash link
            actual_prev_hash = record.get("prev_hash")
            if actual_prev_hash != expected_prev_hash:
                return False, (
                    f"Hash Chain Broken at line {line_number}! "
                    f"Expected prev_hash '{expected_prev_hash[:12]}...', but found '{actual_prev_hash[:12]}...'."
                )

            # 2. Recalculate entry_hash
            payload_dict = {
                "entry_id": record.get("entry_id"),
                "timestamp": record.get("timestamp"),
                "prompt_hash": record.get("prompt_hash"),
                "prev_hash": actual_prev_hash,
                "decision": record.get("decision"),
                "composite_score": record.get("composite_score"),
                "is_financial_trigger": record.get("is_financial_trigger"),
                "trace": record.get("trace")
            }
            canonical_str = json.dumps(payload_dict, sort_keys=True)
            recalculated_hash = _calculate_sha256(f"{actual_prev_hash}:{canonical_str}")

            if recalculated_hash != record.get("entry_hash"):
                return False, (
                    f"Tampered Payload detected at line {line_number}! "
                    f"Computed hash '{recalculated_hash[:12]}...' does not match recorded hash '{record.get('entry_hash')[:12]}...'."
                )

            expected_prev_hash = record.get("entry_hash")

    return True, f"Cryptographic Verification Succeeded: {line_number} audit entries verified with 100% SHA-256 chain continuity."


# ============================================================================
# 2. Human-In-The-Loop (HITL) Queue & Active Feedback Store
# ============================================================================

class HITLQueueManager:
    """Manages quarantined interaction tickets and compliance reviewer sign-offs."""

    def __init__(self):
        self.tickets: Dict[str, HITLTicket] = {}
        self.feedback_records: List[Dict[str, Any]] = []

    def enqueue(
        self,
        prompt: str,
        candidate_response: str,
        arbitration: ArbitrationResult
    ) -> HITLTicket:
        """Creates and enqueues a new quarantined ticket for human review."""
        ticket_id = f"TICKET-{len(self.tickets) + 1:04d}"
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        ticket = HITLTicket(
            ticket_id=ticket_id,
            timestamp=now_iso,
            prompt=prompt,
            candidate_response=candidate_response,
            composite_score=arbitration.composite_score,
            is_financial_trigger=arbitration.is_financial_trigger,
            reason=arbitration.reason,
            status="PENDING"
        )
        self.tickets[ticket_id] = ticket
        return ticket

    def get_ticket(self, ticket_id: str) -> Optional[HITLTicket]:
        return self.tickets.get(ticket_id)

    def list_pending_tickets(self) -> List[HITLTicket]:
        return [t for t in self.tickets.values() if t.status == "PENDING"]

    def resolve_ticket(
        self,
        ticket_id: str,
        action: HITLAction,
        edited_text: Optional[str] = None,
        reviewer_notes: Optional[str] = None
    ) -> HITLTicket:
        """
        Processes human reviewer decision (APPROVE | EDIT | OVERRIDE)
        and feeds outcome to the Active Learning store.
        """
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            raise KeyError(f"Ticket '{ticket_id}' not found.")

        ticket.status = action.value
        ticket.reviewer_notes = reviewer_notes or f"Resolved via action: {action.value}"

        if action == HITLAction.APPROVE:
            ticket.final_delivered_text = ticket.candidate_response
            feedback_type = "TRUE_ALLOW"
        elif action == HITLAction.EDIT:
            ticket.final_delivered_text = edited_text or ticket.candidate_response
            feedback_type = "EDIT_SANITIZED"
        elif action == HITLAction.OVERRIDE:
            ticket.final_delivered_text = (
                edited_text or "This response was blocked by human compliance override."
            )
            feedback_type = "FORCE_BLOCK"

        # Record active learning feedback
        self.feedback_records.append({
            "ticket_id": ticket_id,
            "original_score": ticket.composite_score,
            "action": action.value,
            "feedback_type": feedback_type,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })

        return ticket

    def get_policy_tuning_metrics(self) -> Dict[str, Any]:
        """Calculates calibration feedback metrics for Policy Matrix Optimizer."""
        total = len(self.feedback_records)
        if total == 0:
            return {"total_reviews": 0, "approval_rate": 1.0, "override_rate": 0.0}

        approvals = sum(1 for r in self.feedback_records if r["action"] == HITLAction.APPROVE.value)
        edits = sum(1 for r in self.feedback_records if r["action"] == HITLAction.EDIT.value)
        overrides = sum(1 for r in self.feedback_records if r["action"] == HITLAction.OVERRIDE.value)

        return {
            "total_reviews": total,
            "approval_rate": round(approvals / total, 3),
            "edit_rate": round(edits / total, 3),
            "override_rate": round(overrides / total, 3),
            "recommended_allow_threshold_adjustment": (
                -0.1 if (overrides / total) > 0.3 else 0.05 if (approvals / total) > 0.7 else 0.0
            )
        }


# Global singleton instance for in-memory queue management
hitl_queue_manager = HITLQueueManager()
