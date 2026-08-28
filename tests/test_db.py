"""
test_db.py - Tests for SQLite Persistence Layer (db.py)
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from db import (
    init_db,
    get_all_knowledge_chunks,
    upsert_knowledge_chunk,
    save_hitl_ticket,
    get_hitl_ticket,
    list_pending_hitl_tickets,
    record_feedback,
    get_policy_tuning_metrics_from_db
)
from models import KnowledgeChunk, HITLTicket


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_controlplane.db")
    init_db(db_file)
    return db_file


def test_init_db_seeds_knowledge_base(temp_db):
    chunks = get_all_knowledge_chunks(temp_db)
    assert len(chunks) >= 3
    doc_ids = {c.doc_id for c in chunks}
    assert "KB-001" in doc_ids
    assert "KB-002" in doc_ids
    assert "KB-003" in doc_ids


def test_upsert_knowledge_chunk(temp_db):
    new_chunk = KnowledgeChunk(
        doc_id="KB-999",
        title="New Test Policy",
        category="testing",
        content="This is test policy content.",
        keywords=["test", "policy"]
    )
    upsert_knowledge_chunk(new_chunk, temp_db)
    chunks = get_all_knowledge_chunks(temp_db)
    assert any(c.doc_id == "KB-999" for c in chunks)


def test_save_and_get_hitl_ticket(temp_db):
    ticket = HITLTicket(
        ticket_id="TICKET-TEST-1",
        timestamp="2026-08-28T12:00:00Z",
        prompt="Transfer $5000",
        candidate_response="Draft response",
        composite_score=5.5,
        is_financial_trigger=True,
        reason="Financial trigger",
        status="PENDING"
    )
    save_hitl_ticket(ticket, temp_db)
    
    fetched = get_hitl_ticket("TICKET-TEST-1", temp_db)
    assert fetched is not None
    assert fetched.ticket_id == "TICKET-TEST-1"
    assert fetched.is_financial_trigger is True
    assert fetched.composite_score == 5.5


def test_list_pending_hitl_tickets(temp_db):
    t1 = HITLTicket(
        ticket_id="TICKET-P1",
        timestamp="2026-08-28T12:01:00Z",
        prompt="P1",
        candidate_response="R1",
        composite_score=4.0,
        is_financial_trigger=False,
        reason="Score",
        status="PENDING"
    )
    t2 = HITLTicket(
        ticket_id="TICKET-P2",
        timestamp="2026-08-28T12:02:00Z",
        prompt="P2",
        candidate_response="R2",
        composite_score=3.0,
        is_financial_trigger=False,
        reason="Score",
        status="APPROVED"
    )
    save_hitl_ticket(t1, temp_db)
    save_hitl_ticket(t2, temp_db)

    pending = list_pending_hitl_tickets(temp_db)
    assert len(pending) == 1
    assert pending[0].ticket_id == "TICKET-P1"


def test_record_feedback_and_metrics(temp_db):
    record_feedback("TICKET-1", 5.0, "APPROVE", "TRUE_ALLOW", "2026-08-28T12:00:00Z", temp_db)
    record_feedback("TICKET-2", 4.5, "APPROVE", "TRUE_ALLOW", "2026-08-28T12:01:00Z", temp_db)
    record_feedback("TICKET-3", 6.0, "OVERRIDE", "FORCE_BLOCK", "2026-08-28T12:02:00Z", temp_db)

    metrics = get_policy_tuning_metrics_from_db(temp_db)
    assert metrics["total_reviews"] == 3
    assert metrics["approval_rate"] == 0.667
    assert metrics["override_rate"] == 0.333
