"""
test_rag_verifier.py - Tests for Stage 3B: RAG Grounding Verification
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from rag_verifier import (
    retrieve_knowledge_chunks,
    verify_factual_grounding,
    _extract_numeric_entities,
    _normalize_token,
)
from models import KnowledgeChunk, ENTERPRISE_KNOWLEDGE_BASE


# ============================================================================
# Knowledge Chunk Retrieval Tests
# ============================================================================

class TestRetrieveKnowledgeChunks:
    """Tests for the enterprise RAG retriever."""

    def test_refund_query_retrieves_refund_policy(self):
        chunks = retrieve_knowledge_chunks("What is the refund policy?")
        assert len(chunks) >= 1
        assert any("refund" in c.category.lower() for c in chunks)

    def test_credit_query_retrieves_underwriting(self):
        chunks = retrieve_knowledge_chunks("credit line increase loan approval")
        assert len(chunks) >= 1
        assert any("credit" in c.category.lower() for c in chunks)

    def test_unrelated_query_returns_empty(self):
        """General greetings should not match any knowledge chunks."""
        chunks = retrieve_knowledge_chunks("Hello, how are you today?")
        assert len(chunks) == 0

    def test_top_k_limits_results(self):
        chunks = retrieve_knowledge_chunks("refund policy credit loan", top_k=1)
        assert len(chunks) <= 1

    def test_custom_knowledge_base(self):
        """Tests passing a custom KB instead of the default."""
        custom_kb = [
            KnowledgeChunk(
                doc_id="CUSTOM-001",
                title="Test Policy",
                category="test",
                content="Custom content for testing retrieval.",
                keywords=["custom", "test"]
            )
        ]
        chunks = retrieve_knowledge_chunks("custom test query", kb=custom_kb)
        assert len(chunks) == 1
        assert chunks[0].doc_id == "CUSTOM-001"

    def test_default_kb_used_when_none(self):
        """Verifying None default resolves to ENTERPRISE_KNOWLEDGE_BASE."""
        chunks = retrieve_knowledge_chunks("return refund 30 days", kb=None)
        assert len(chunks) >= 1


# ============================================================================
# Numeric Entity Extraction Tests
# ============================================================================

class TestNumericExtraction:
    """Tests for numeric, currency, and timeframe extraction."""

    def test_dollar_amount_extracted(self):
        entities = _extract_numeric_entities("Refund up to $500")
        assert any("$500" in e for e in entities)

    def test_timeframe_extracted(self):
        entities = _extract_numeric_entities("Return within 30 days")
        assert any("30 days" in e for e in entities)

    def test_percentage_extracted(self):
        entities = _extract_numeric_entities("Interest rate is 5.5 percent")
        assert any("5.5 percent" in e for e in entities)

    def test_business_days_extracted(self):
        entities = _extract_numeric_entities("Processed within 7 business days")
        assert any("7 business days" in e for e in entities)

    def test_no_entities_in_clean_text(self):
        entities = _extract_numeric_entities("Hello world, how are you?")
        assert len(entities) == 0


# ============================================================================
# Grounding Verification Tests
# ============================================================================

class TestGroundingVerification:
    """Tests for factual grounding score calculation."""

    def test_grounded_response_scores_high(self):
        """Response matching KB facts should have high grounding score."""
        result = verify_factual_grounding(
            candidate_response="Standard items can be returned within 30 days for a full refund.",
            query="What is the return policy?"
        )
        assert result.grounding_score >= 7.0
        assert result.rag_risk <= 3.0

    def test_fabricated_number_detected(self):
        """Numbers not in KB should be flagged as mismatches."""
        result = verify_factual_grounding(
            candidate_response="You can return items within 90 days for a $500 cash refund.",
            query="What is the return policy?"
        )
        assert len(result.numeric_mismatches) > 0
        assert result.grounding_score < 10.0

    def test_no_kb_match_returns_clean(self):
        """General queries with no KB match should default to grounding_score=10."""
        result = verify_factual_grounding(
            candidate_response="Hello! How can I help you today?",
            query="Hi there"
        )
        assert result.grounding_score == 10.0
        assert result.rag_risk == 0.0
        assert len(result.retrieved_chunks) == 0

    def test_penalty_capped_at_zero_score(self):
        """Multiple mismatches should not push score below 0."""
        result = verify_factual_grounding(
            candidate_response="Refund $999 in 120 days with 75% bonus and $5000 credit after 365 days.",
            query="What is the refund and return policy?"
        )
        assert result.grounding_score >= 0.0
        assert result.rag_risk <= 10.0


# ============================================================================
# Normalize Token Tests
# ============================================================================

class TestNormalizeToken:
    """Tests for token normalization."""

    def test_basic_normalization(self):
        assert _normalize_token("  $500  ") == "$500"

    def test_whitespace_collapsed(self):
        assert _normalize_token("30   days") == "30 days"

    def test_case_lowered(self):
        assert _normalize_token("REFUND") == "refund"
