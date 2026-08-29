"""
test_audit_hitl.py - Tests for Stage 5: Audit Logging & HITL Queue Management
"""

import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from audit_hitl import (
    log_audit_entry,
    get_latest_audit_hash,
    verify_audit_log_integrity,
    HITLQueueManager,
    GENESIS_HASH,
    _calculate_sha256,
)
from models import ArbitrationResult, DecisionTier, HITLAction


# ============================================================================
# SHA-256 Hash Chain Tests
# ============================================================================

class TestAuditHashChain:
    """Tests for immutable SHA-256 hash-chained audit logging."""

    @pytest.fixture
    def temp_log(self, tmp_path):
        """Creates a temporary log file path for isolated testing."""
        return str(tmp_path / "test_audit.jsonl")

    def test_genesis_hash_on_empty_file(self, temp_log):
        """Empty/nonexistent file should return the genesis hash."""
        assert get_latest_audit_hash(temp_log) == GENESIS_HASH

    def test_single_entry_creates_chain(self, temp_log):
        arb = ArbitrationResult(
            composite_score=1.0,
            decision=DecisionTier.ALLOW,
            is_financial_trigger=False
        )
        entry = log_audit_entry("test prompt", arb, {"test": True}, log_path=temp_log)
        
        assert entry.prev_hash == GENESIS_HASH
        assert entry.entry_hash != GENESIS_HASH
        assert entry.entry_id.startswith("AUDIT-")

    def test_chain_continuity_across_entries(self, temp_log):
        arb = ArbitrationResult(
            composite_score=2.0,
            decision=DecisionTier.ALLOW,
            is_financial_trigger=False
        )
        entry1 = log_audit_entry("prompt 1", arb, {"step": 1}, log_path=temp_log)
        entry2 = log_audit_entry("prompt 2", arb, {"step": 2}, log_path=temp_log)

        # Entry 2's prev_hash should equal Entry 1's entry_hash
        assert entry2.prev_hash == entry1.entry_hash

    def test_integrity_passes_on_valid_log(self, temp_log):
        arb = ArbitrationResult(
            composite_score=1.0,
            decision=DecisionTier.ALLOW,
            is_financial_trigger=False
        )
        for i in range(5):
            log_audit_entry(f"prompt {i}", arb, {"step": i}, log_path=temp_log)

        is_valid, msg = verify_audit_log_integrity(temp_log)
        assert is_valid is True
        assert "5" in msg  # Should mention 5 entries verified

    def test_integrity_detects_tampering(self, temp_log):
        arb = ArbitrationResult(
            composite_score=1.0,
            decision=DecisionTier.ALLOW,
            is_financial_trigger=False
        )
        for i in range(3):
            log_audit_entry(f"prompt {i}", arb, {"step": i}, log_path=temp_log)

        # Tamper with the second line
        with open(temp_log, "r") as f:
            lines = f.readlines()
        
        tampered = json.loads(lines[1])
        tampered["composite_score"] = 999.0
        lines[1] = json.dumps(tampered) + "\n"
        
        with open(temp_log, "w") as f:
            f.writelines(lines)

        is_valid, msg = verify_audit_log_integrity(temp_log)
        assert is_valid is False
        assert "tamper" in msg.lower() or "hash" in msg.lower()

    def test_uuid_based_entry_ids_are_unique(self, temp_log):
        arb = ArbitrationResult(
            composite_score=1.0,
            decision=DecisionTier.ALLOW,
            is_financial_trigger=False
        )
        ids = set()
        for i in range(10):
            entry = log_audit_entry(f"prompt {i}", arb, {"step": i}, log_path=temp_log)
            ids.add(entry.entry_id)
        
        assert len(ids) == 10  # All IDs should be unique

    def test_latest_hash_reads_last_entry(self, temp_log):
        arb = ArbitrationResult(
            composite_score=1.0,
            decision=DecisionTier.ALLOW,
            is_financial_trigger=False
        )
        entries = []
        for i in range(3):
            entries.append(log_audit_entry(f"p{i}", arb, {}, log_path=temp_log))
        
        latest = get_latest_audit_hash(temp_log)
        assert latest == entries[-1].entry_hash


# ============================================================================
# HITL Queue Management Tests
# ============================================================================

class TestHITLQueueManager:
    """Tests for HITL ticket lifecycle management."""

    @pytest.fixture
    def manager(self):
        return HITLQueueManager()

    @pytest.fixture
    def sample_arbitration(self):
        return ArbitrationResult(
            composite_score=5.0,
            decision=DecisionTier.HITL,
            is_financial_trigger=True,
            reason="Financial trigger detected"
        )

    def test_enqueue_creates_ticket(self, manager, sample_arbitration):
        ticket = manager.enqueue("test prompt", "test response", sample_arbitration)
        assert ticket.ticket_id.startswith("TICKET-")
        assert ticket.status == "PENDING"
        assert ticket.composite_score == 5.0

    def test_ticket_ids_are_unique(self, manager, sample_arbitration):
        ids = set()
        for _ in range(20):
            ticket = manager.enqueue("p", "r", sample_arbitration)
            ids.add(ticket.ticket_id)
        assert len(ids) == 20

    def test_list_pending_tickets(self, manager, sample_arbitration):
        manager.enqueue("p1", "r1", sample_arbitration)
        manager.enqueue("p2", "r2", sample_arbitration)
        pending = manager.list_pending_tickets()
        assert len(pending) == 2

    def test_approve_resolves_ticket(self, manager, sample_arbitration):
        ticket = manager.enqueue("prompt", "candidate response", sample_arbitration)
        resolved = manager.resolve_ticket(ticket.ticket_id, HITLAction.ALLOW)
        
        assert resolved.status in ("ALLOW", "APPROVE")
        assert resolved.final_delivered_text == "candidate response"

    def test_edit_resolves_with_custom_text(self, manager, sample_arbitration):
        ticket = manager.enqueue("prompt", "original", sample_arbitration)
        resolved = manager.resolve_ticket(
            ticket.ticket_id,
            HITLAction.EDIT,
            edited_text="sanitized version"
        )
        assert resolved.status == "EDIT"
        assert resolved.final_delivered_text == "sanitized version"

    def test_override_blocks_response(self, manager, sample_arbitration):
        ticket = manager.enqueue("prompt", "original", sample_arbitration)
        resolved = manager.resolve_ticket(ticket.ticket_id, HITLAction.BLOCK)
        assert resolved.status in ("BLOCK", "OVERRIDE")
        assert "blocked" in resolved.final_delivered_text.lower()

    def test_resolve_nonexistent_ticket_raises(self, manager):
        with pytest.raises(KeyError):
            manager.resolve_ticket("TICKET-NONEXISTENT", HITLAction.APPROVE)

    def test_pending_count_decreases_after_resolve(self, manager, sample_arbitration):
        t1 = manager.enqueue("p1", "r1", sample_arbitration)
        manager.enqueue("p2", "r2", sample_arbitration)
        
        assert len(manager.list_pending_tickets()) == 2
        manager.resolve_ticket(t1.ticket_id, HITLAction.APPROVE)
        assert len(manager.list_pending_tickets()) == 1


# ============================================================================
# Policy Tuning Metrics Tests
# ============================================================================

class TestPolicyTuningMetrics:
    """Tests for active feedback store calibration metrics."""

    @pytest.fixture
    def manager_with_feedback(self):
        mgr = HITLQueueManager()
        arb = ArbitrationResult(
            composite_score=5.0,
            decision=DecisionTier.HITL,
            is_financial_trigger=False
        )
        # Create and resolve 4 tickets with different actions
        for _ in range(3):
            t = mgr.enqueue("p", "r", arb)
            mgr.resolve_ticket(t.ticket_id, HITLAction.APPROVE)
        t = mgr.enqueue("p", "r", arb)
        mgr.resolve_ticket(t.ticket_id, HITLAction.OVERRIDE)
        return mgr

    def test_metrics_correct_counts(self, manager_with_feedback):
        metrics = manager_with_feedback.get_policy_tuning_metrics()
        assert metrics["total_reviews"] == 4
        assert metrics["approval_rate"] == 0.75
        assert metrics["override_rate"] == 0.25

    def test_empty_manager_defaults(self):
        mgr = HITLQueueManager()
        metrics = mgr.get_policy_tuning_metrics()
        assert metrics["total_reviews"] == 0
        assert metrics["approval_rate"] == 1.0
