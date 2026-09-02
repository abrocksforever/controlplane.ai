"""
db.py - SQLite Persistence Layer for ControlPlane.ai
ControlPlane.ai (PS1 Architecture)

Provides a lightweight, zero-dependency persistence layer using Python built-in sqlite3:
1. Persistent HITL Triage Queue (hitl_tickets)
2. Dynamic Enterprise Policy Store (knowledge_base with version 2 schema)
3. Active Learning Feedback Store (feedback_store)
"""

import os
import re
import glob
import json
import sqlite3
import logging
import datetime
from typing import List, Dict, Any, Optional

from models import KnowledgeChunk, HITLTicket

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "controlplane.db"

# English Stopwords to prevent BM25 IDF dilution
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "you", "your", "can", "our", "all", "any", "how", "what",
    "when", "where", "which", "who", "why", "does", "have", "this", "or", "but"
}


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Returns a thread-safe sqlite3 database connection with Row factory."""
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def parse_markdown_policy_file(file_path: str) -> Optional[KnowledgeChunk]:
    """
    Safely extracts YAML frontmatter and Markdown body with fallback validation.
    Filters stopwords out of automatically extracted keywords.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()

        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.DOTALL)
        if match:
            meta_block, content = match.group(1), match.group(2).strip()
            meta = {}
            for line in meta_block.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip("'\"")
        else:
            meta = {}
            content = raw.strip()

        doc_id = meta.get("document_id", os.path.splitext(os.path.basename(file_path))[0])
        title = meta.get("title", doc_id.replace("_", " ").title())
        category = meta.get("category", "general")
        product = meta.get("product", "all")
        audience = meta.get("audience", "guest")
        region = meta.get("region", "global")
        source_url = meta.get("source_url", f"https://www.airbnb.com/help/article/{doc_id}")

        # Extract content keywords excluding stopwords
        tokens = re.findall(r"\b[a-z0-9_$-]{3,}\b", (title + " " + content).lower())
        keywords = [t for t in tokens if t not in STOPWORDS][:30]

        return KnowledgeChunk(
            doc_id=doc_id,
            title=title,
            category=category,
            product=product,
            audience=audience,
            region=region,
            source_url=source_url,
            content=content,
            keywords=keywords
        )
    except Exception as e:
        logger.error(f"Failed parsing markdown policy '{file_path}': {e}")
        return None


def load_airbnb_corpus_chunks(corpus_dir: Optional[str] = None) -> List[KnowledgeChunk]:
    """Loads and parses all 20 Markdown policy files from the Airbnb cleaned directory."""
    if not corpus_dir:
        # Check standard relative paths
        candidates = [
            "airbnb-grounding-rag-kb/cleaned",
            os.path.join(os.path.dirname(__file__), "airbnb-grounding-rag-kb", "cleaned"),
            os.path.join(os.getcwd(), "airbnb-grounding-rag-kb", "cleaned")
        ]
        for c in candidates:
            if os.path.exists(c):
                corpus_dir = c
                break

    chunks = []
    if corpus_dir and os.path.exists(corpus_dir):
        files = glob.glob(os.path.join(corpus_dir, "**", "*.md"), recursive=True)
        for f in sorted(files):
            chunk = parse_markdown_policy_file(f)
            if chunk:
                chunks.append(chunk)

    return chunks


_INITIALIZED_DBS = set()


def init_db(db_path: str = DEFAULT_DB_PATH, force: bool = False) -> None:
    """
    Initializes database tables with schema versioning (PRAGMA user_version = 2)
    and seeds all 20 Airbnb knowledge base documents.
    Fast-returns if already initialized for this process.
    """
    global _INITIALIZED_DBS
    if not force and db_path in _INITIALIZED_DBS and os.path.exists(db_path):
        return

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Check Schema Version
        cursor.execute("PRAGMA user_version;")
        version_row = cursor.fetchone()
        current_version = version_row[0] if version_row else 0

        if current_version < 2:
            logger.info(f"Migrating database '{db_path}' to schema version 2...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base_v2 (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    product TEXT NOT NULL DEFAULT 'all',
                    audience TEXT NOT NULL DEFAULT 'guest',
                    region TEXT NOT NULL DEFAULT 'global',
                    source_url TEXT,
                    content TEXT NOT NULL,
                    keywords TEXT NOT NULL
                );
            """)

            # Check if old table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_base';")
            if cursor.fetchone():
                # Purge obsolete toy chunks (KB-001..003) and migrate non-toy data
                cursor.execute("""
                    INSERT OR IGNORE INTO knowledge_base_v2 (doc_id, title, category, content, keywords)
                    SELECT doc_id, title, category, content, keywords FROM knowledge_base
                    WHERE doc_id NOT LIKE 'KB-%';
                """)
                cursor.execute("DROP TABLE knowledge_base;")

            cursor.execute("ALTER TABLE knowledge_base_v2 RENAME TO knowledge_base;")
            cursor.execute("PRAGMA user_version = 2;")
            conn.commit()

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

        # 4. Interaction History Table (All conversations: ALLOW, HITL, BLOCK)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                decision TEXT NOT NULL,
                composite_score REAL NOT NULL,
                final_response TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                audit_hash TEXT NOT NULL
            );
        """)

        # Seed 20 Airbnb Knowledge Chunks
        cursor.execute("SELECT COUNT(*) as cnt FROM knowledge_base;")
        row = cursor.fetchone()
        if row and row["cnt"] == 0:
            airbnb_chunks = load_airbnb_corpus_chunks()
            if airbnb_chunks:
                for chunk in airbnb_chunks:
                    cursor.execute("""
                        INSERT OR REPLACE INTO knowledge_base (
                            doc_id, title, category, product, audience, region, source_url, content, keywords
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (
                        chunk.doc_id,
                        chunk.title,
                        chunk.category,
                        chunk.product or "all",
                        chunk.audience or "guest",
                        chunk.region or "global",
                        chunk.source_url or "",
                        chunk.content,
                        json.dumps(chunk.keywords)
                    ))
                logger.info(f"Seeded {len(airbnb_chunks)} Airbnb knowledge chunks into '{db_path}'.")

        conn.commit()
        _INITIALIZED_DBS.add(db_path)


# ============================================================================
# Knowledge Base Operations (with In-Memory Caching for <0.05ms Retrieval)
# ============================================================================

_KB_CACHE: Optional[List[KnowledgeChunk]] = None
_KB_CACHE_PATH: Optional[str] = None


def get_all_knowledge_chunks(db_path: str = DEFAULT_DB_PATH) -> List[KnowledgeChunk]:
    """Loads all knowledge chunks from the database (in-memory cached after first load)."""
    global _KB_CACHE, _KB_CACHE_PATH
    if _KB_CACHE is not None and _KB_CACHE_PATH == db_path:
        return _KB_CACHE

    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT doc_id, title, category, product, audience, region, source_url, content, keywords
            FROM knowledge_base;
        """)
        rows = cursor.fetchall()
        
        chunks = []
        for r in rows:
            kw = json.loads(r["keywords"]) if r["keywords"] else []
            chunks.append(
                KnowledgeChunk(
                    doc_id=r["doc_id"],
                    title=r["title"],
                    category=r["category"],
                    product=r["product"] if "product" in r.keys() else "all",
                    audience=r["audience"] if "audience" in r.keys() else "guest",
                    region=r["region"] if "region" in r.keys() else "global",
                    source_url=r["source_url"] if "source_url" in r.keys() else None,
                    content=r["content"],
                    keywords=kw
                )
            )
        _KB_CACHE = chunks
        _KB_CACHE_PATH = db_path
        return chunks


def upsert_knowledge_chunk(chunk: KnowledgeChunk, db_path: str = DEFAULT_DB_PATH) -> None:
    """Inserts or updates an enterprise policy knowledge chunk and invalidates in-memory cache."""
    global _KB_CACHE
    _KB_CACHE = None  # Invalidate cache
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO knowledge_base (
                doc_id, title, category, product, audience, region, source_url, content, keywords
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                title=excluded.title,
                category=excluded.category,
                product=excluded.product,
                audience=excluded.audience,
                region=excluded.region,
                source_url=excluded.source_url,
                content=excluded.content,
                keywords=excluded.keywords;
        """, (
            chunk.doc_id,
            chunk.title,
            chunk.category,
            chunk.product or "all",
            chunk.audience or "guest",
            chunk.region or "global",
            chunk.source_url or "",
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


def list_all_hitl_tickets(limit: int = 500, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves all recent tickets up to limit from the database."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM hitl_tickets ORDER BY timestamp DESC LIMIT ?;", (limit,))
        return [dict(r) for r in cursor.fetchall()]


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
            return {
                "total_reviews": 0,
                "allow_rate": 1.0,
                "approval_rate": 1.0,
                "edit_rate": 0.0,
                "block_rate": 0.0,
                "override_rate": 0.0
            }

        allows = sum(1 for r in rows if r["action"] in ("ALLOW", "APPROVE"))
        edits = sum(1 for r in rows if r["action"] == "EDIT")
        blocks = sum(1 for r in rows if r["action"] in ("BLOCK", "OVERRIDE"))

        allow_rate = round(allows / total, 3)
        edit_rate = round(edits / total, 3)
        block_rate = round(blocks / total, 3)

        return {
            "total_reviews": total,
            "allow_rate": allow_rate,
            "approval_rate": allow_rate,
            "edit_rate": edit_rate,
            "block_rate": block_rate,
            "override_rate": block_rate,
            "recommended_allow_threshold_adjustment": (
                -0.1 if (blocks / total) > 0.3 else 0.05 if (allows / total) > 0.7 else 0.0
            )
        }


# ============================================================================
# Interaction History Operations (All Conversations)
# ============================================================================

def save_interaction(
    trace_id: str,
    prompt: str,
    decision: str,
    composite_score: float,
    final_response: str,
    latency_ms: float,
    audit_hash: str,
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """Saves every incoming conversation prompt, decision, and response to SQLite."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cursor.execute("""
            INSERT INTO interactions (timestamp, trace_id, prompt, decision, composite_score, final_response, latency_ms, audit_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, (now_iso, trace_id, prompt, decision, composite_score, final_response, latency_ms, audit_hash))
        conn.commit()


def list_interactions(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves recent conversation interactions from SQLite."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, trace_id, prompt, decision, composite_score, final_response, latency_ms, audit_hash
            FROM interactions ORDER BY id DESC LIMIT ?;
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def reset_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Completely resets the SQLite database:
    1. Removes existing database file if present.
    2. Initializes clean schema version 2 tables.
    3. Re-seeds all 20 authoritative Airbnb knowledge base documents.
    """
    global _KB_CACHE
    _KB_CACHE = None  # Invalidate in-memory cache

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            logger.info(f"Removed existing database file '{db_path}'.")
        except Exception as e:
            logger.warning(f"Could not remove '{db_path}', wiping tables via SQL: {e}")
            with get_connection(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DROP TABLE IF EXISTS knowledge_base;")
                cursor.execute("DROP TABLE IF EXISTS hitl_tickets;")
                cursor.execute("DROP TABLE IF EXISTS feedback_store;")
                cursor.execute("PRAGMA user_version = 0;")
                conn.commit()

    init_db(db_path)
    chunks = get_all_knowledge_chunks(db_path)
    print(f"Database '{db_path}' successfully reset and seeded with {len(chunks)} Airbnb knowledge chunks.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ControlPlane.ai SQLite Database Utility")
    parser.add_argument("--reset", action="store_true", help="Reset database and re-seed all 20 Airbnb documents")
    parser.add_argument("--status", action="store_true", help="Display current database record counts")
    args = parser.parse_args()

    if args.reset:
        reset_db()
    elif args.status:
        init_db()
        chunks = get_all_knowledge_chunks()
        pending = list_pending_hitl_tickets()
        metrics = get_policy_tuning_metrics_from_db()
        interactions = list_interactions(limit=1000)
        print(f"Database Status (controlplane.db):")
        print(f"  - Knowledge Base Documents:    {len(chunks)}")
        print(f"  - Total Recorded Interactions: {len(interactions)}")
        print(f"  - Pending HITL Tickets:        {len(pending)}")
        print(f"  - Total Feedback Reviews:      {metrics['total_reviews']}")
    else:
        init_db()
        print("Database initialized. Use --reset to re-seed or --status to view counts.")
