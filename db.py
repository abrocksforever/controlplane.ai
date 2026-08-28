"""
db.py - SQLite Persistence Layer for ControlPlane.ai
ControlPlane.ai (PS1 Architecture)

Provides a lightweight, zero-dependency persistence layer using Python built-in sqlite3:
1. Persistent HITL Triage Queue (hitl_tickets)
2. Dynamic Enterprise Policy Store (knowledge_base)
3. Active Learning Feedback Store (feedback_store)
"""

import os
import json
import sqlite3
import logging
from typing import List, Dict, Any, Optional

from models import KnowledgeChunk, HITLTicket, ENTERPRISE_KNOWLEDGE_BASE

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "controlplane.db"


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Returns a thread-safe sqlite3 database connection with Row factory."""
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initializes database tables and seeds default enterprise knowledge base chunks.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Knowledge Base Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                doc_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                keywords TEXT NOT NULL
            );
        """)

        # 2. HITL Review Queue Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hitl_tickets (
                ticket_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                prompt TEXT NOT NULL,
                candidate_response TEXT NOT NULL,
                composite_score REAL NOT NULL,
                is_financial_trigger INTEGER NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                reviewer_notes TEXT,
                final_delivered_text TEXT
            );
        """)

        # 3. Active Learning Feedback Store Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                original_score REAL NOT NULL,
                action TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
        """)

        # Seed initial knowledge chunks if knowledge_base is empty
        cursor.execute("SELECT COUNT(*) as cnt FROM knowledge_base;")
        row = cursor.fetchone()
        if row and row["cnt"] == 0:
            for chunk in ENTERPRISE_KNOWLEDGE_BASE:
                cursor.execute("""
                    INSERT INTO knowledge_base (doc_id, title, category, content, keywords)
                    VALUES (?, ?, ?, ?, ?);
                """, (
                    chunk.doc_id,
                    chunk.title,
                    chunk.category,
                    chunk.content,
                    json.dumps(chunk.keywords)
                ))
            logger.info(f"Seeded {len(ENTERPRISE_KNOWLEDGE_BASE)} knowledge chunks into database.")

        conn.commit()


# ============================================================================
# Knowledge Base Operations
# ============================================================================

def get_all_knowledge_chunks(db_path: str = DEFAULT_DB_PATH) -> List[KnowledgeChunk]:
    """Loads all knowledge chunks from the database."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, title, category, content, keywords FROM knowledge_base;")
        rows = cursor.fetchall()
        
        chunks = []
        for r in rows:
            kw = json.loads(r["keywords"]) if r["keywords"] else []
            chunks.append(
                KnowledgeChunk(
                    doc_id=r["doc_id"],
                    title=r["title"],
                    category=r["category"],
                    content=r["content"],
                    keywords=kw
                )
            )
        return chunks


def upsert_knowledge_chunk(chunk: KnowledgeChunk, db_path: str = DEFAULT_DB_PATH) -> None:
    """Inserts or updates an enterprise policy knowledge chunk."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO knowledge_base (doc_id, title, category, content, keywords)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                title=excluded.title,
                category=excluded.category,
                content=excluded.content,
                keywords=excluded.keywords;
        """, (
            chunk.doc_id,
            chunk.title,
            chunk.category,
            chunk.content,
            json.dumps(chunk.keywords)
        ))
        conn.commit()


# ============================================================================
# HITL Queue Operations
# ============================================================================

def save_hitl_ticket(ticket: HITLTicket, db_path: str = DEFAULT_DB_PATH) -> None:
    """Saves or updates a HITL ticket in the database."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO hitl_tickets (
                ticket_id, timestamp, prompt, candidate_response, composite_score,
                is_financial_trigger, reason, status, reviewer_notes, final_delivered_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticket_id) DO UPDATE SET
                status=excluded.status,
                reviewer_notes=excluded.reviewer_notes,
                final_delivered_text=excluded.final_delivered_text;
        """, (
            ticket.ticket_id,
            ticket.timestamp,
            ticket.prompt,
            ticket.candidate_response,
            ticket.composite_score,
            1 if ticket.is_financial_trigger else 0,
            ticket.reason,
            ticket.status,
            ticket.reviewer_notes,
            ticket.final_delivered_text
        ))
        conn.commit()


def get_hitl_ticket(ticket_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[HITLTicket]:
    """Fetches a specific HITL ticket by ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM hitl_tickets WHERE ticket_id = ?;", (ticket_id,))
        row = cursor.fetchone()
        if not row:
            return None
            
        return HITLTicket(
            ticket_id=row["ticket_id"],
            timestamp=row["timestamp"],
            prompt=row["prompt"],
            candidate_response=row["candidate_response"],
            composite_score=row["composite_score"],
            is_financial_trigger=bool(row["is_financial_trigger"]),
            reason=row["reason"],
            status=row["status"],
            reviewer_notes=row["reviewer_notes"],
            final_delivered_text=row["final_delivered_text"]
        )


def list_pending_hitl_tickets(db_path: str = DEFAULT_DB_PATH) -> List[HITLTicket]:
    """Retrieves all pending tickets from the database."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM hitl_tickets WHERE status = 'PENDING' ORDER BY timestamp DESC;")
        rows = cursor.fetchall()
        
        return [
            HITLTicket(
                ticket_id=r["ticket_id"],
                timestamp=r["timestamp"],
                prompt=r["prompt"],
                candidate_response=r["candidate_response"],
                composite_score=r["composite_score"],
                is_financial_trigger=bool(r["is_financial_trigger"]),
                reason=r["reason"],
                status=r["status"],
                reviewer_notes=r["reviewer_notes"],
                final_delivered_text=r["final_delivered_text"]
            )
            for r in rows
        ]


# ============================================================================
# Active Learning Feedback Operations
# ============================================================================

def record_feedback(
    ticket_id: str,
    original_score: float,
    action: str,
    feedback_type: str,
    timestamp: str,
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """Records reviewer triage action into the feedback store."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback_store (ticket_id, original_score, action, feedback_type, timestamp)
            VALUES (?, ?, ?, ?, ?);
        """, (ticket_id, original_score, action, feedback_type, timestamp))
        conn.commit()


def get_policy_tuning_metrics_from_db(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Calculates active learning calibration metrics directly from database feedback."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT action FROM feedback_store;")
        rows = cursor.fetchall()
        
        total = len(rows)
        if total == 0:
            return {"total_reviews": 0, "approval_rate": 1.0, "override_rate": 0.0}

        approvals = sum(1 for r in rows if r["action"] == "APPROVE")
        edits = sum(1 for r in rows if r["action"] == "EDIT")
        overrides = sum(1 for r in rows if r["action"] == "OVERRIDE")

        return {
            "total_reviews": total,
            "approval_rate": round(approvals / total, 3),
            "edit_rate": round(edits / total, 3),
            "override_rate": round(overrides / total, 3),
            "recommended_allow_threshold_adjustment": (
                -0.1 if (overrides / total) > 0.3 else 0.05 if (approvals / total) > 0.7 else 0.0
            )
        }
