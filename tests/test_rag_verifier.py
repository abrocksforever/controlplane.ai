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
    evaluate_factual_assertions
)
from models import KnowledgeChunk, VerificationStatus


# ============================================================================
# Knowledge Chunk Retrieval Tests
# ============================================================================

class TestRetrieveKnowledgeChunks:
    """Tests for the canonical BM25 RAG retriever."""

    def test_refund_query_retrieves_refund_policy(self):
        chunks = retrieve_knowledge_chunks("What is the refund policy for home cancellations?")
        assert len(chunks) >= 1
        doc_ids = [c.doc_id for c in chunks]
        assert any("refund" in d or "cancellation" in d for d in doc_ids)

    def test_india_upi_query_retrieves_india_refunds(self):
        chunks = retrieve_knowledge_chunks("How long does UPI refund take in India?")
        assert len(chunks) >= 1
        doc_ids = [c.doc_id for c in chunks]
        assert "india_refunds" in doc_ids or "india_payments" in doc_ids

    def test_service_experience_query_retrieves_services(self):
        chunks = retrieve_knowledge_chunks("Experience cancellation policy 72 hours")
        assert len(chunks) >= 1
        doc_ids = [c.doc_id for c in chunks]
        assert any("service" in d for d in doc_ids)

    def test_unrelated_query_returns_empty(self):
        """General greetings should not match any knowledge chunks."""
        chunks = retrieve_knowledge_chunks("Hello, how are you today?")
        assert len(chunks) == 0

    def test_top_k_limits_results(self):
        chunks = retrieve_knowledge_chunks("cancellation refund payment stay", top_k=2)
        assert len(chunks) <= 2

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


# ============================================================================
# Numeric & Timeframe Entity Extraction Tests
# ============================================================================

class TestNumericExtraction:
    """Tests for numeric, currency, and timeframe extraction."""

    def test_dollar_amount_extracted(self):
        entities = _extract_numeric_entities("Refund up to $500")
        assert any("$500" in e for e in entities)

    def test_hours_timeframe_extracted(self):
        entities = _extract_numeric_entities("Cancel until 24 hours before check-in")
        assert any("24 hours" in e for e in entities)

    def test_nights_timeframe_extracted(self):
        entities = _extract_numeric_entities("Monthly stays of 28 nights or more")
        assert any("28 nights" in e for e in entities)

    def test_business_days_extracted(self):
        entities = _extract_numeric_entities("Processed within 15 business days")
        assert any("15 business days" in e for e in entities)

    def test_trailing_punctuation_stripped(self):
        entities = _extract_numeric_entities("Refund is $100, within 30 days.")
        assert "$100" in entities
        assert "30 days" in entities

    def test_no_entities_in_clean_text(self):
        entities = _extract_numeric_entities("Hello world, how are you?")
        assert len(entities) == 0


# ============================================================================
# Grounding Verification Tests
# ============================================================================

class TestGroundingVerification:
    """Tests for factual grounding score and confidence calculation."""

    def test_general_conversation_confidence(self):
        """Greetings with 0 docs should be GENERAL_CONVERSATION with confidence=1.0."""
        result = verify_factual_grounding(
            candidate_response="Hello! How can I help you with your reservation today?",
            query="Hi there",
            knowledge_base=[]
        )
        assert result.verification_status == VerificationStatus.GENERAL_CONVERSATION
        assert result.verification_confidence == 1.0
        assert result.grounding_score == 10.0
        assert result.rag_risk == 0.0

    def test_unverified_assertion_quarantined(self):
        """Policy claims with 0 docs should be UNVERIFIED_ASSERTION with confidence=0.0 and rag_risk=7.0."""
        result = verify_factual_grounding(
            candidate_response="Your cancellation refund will take 90 days and costs $500.",
            query="Can I cancel my trip?",
            knowledge_base=[]
        )
        assert result.verification_status == VerificationStatus.UNVERIFIED_ASSERTION
        assert result.verification_confidence == 0.0
        assert result.grounding_score == 3.0
        assert result.rag_risk == 7.0

    def test_grounded_response_scores_high(self):
        """Response matching India refund policy should have high grounding score."""
        result = verify_factual_grounding(
            candidate_response="Refunds for UPI payments in India typically take up to 15 business days.",
            query="How long does UPI refund take in India?",
            use_nli=False
        )
        assert result.grounding_score >= 7.0
        assert result.verification_confidence == 1.0
        assert result.verification_status == VerificationStatus.VERIFIED_GROUNDED

    def test_fabricated_number_detected(self):
        """Numbers not in policy should be flagged as mismatches."""
        result = verify_factual_grounding(
            candidate_response="UPI refunds in India are guaranteed to arrive in exactly 99 days.",
            query="What is the refund timeline for UPI in India?",
            use_nli=False
        )
        assert len(result.numeric_mismatches) > 0
        assert result.grounding_score < 10.0

    def test_absolute_universal_guarantee_flagged(self):
        """Universal guarantees contradicting listing policies should be heavily penalized."""
        result = verify_factual_grounding(
            candidate_response="Airbnb guarantees that every home reservation provides a 100% full refund regardless of listing policy.",
            query="Can I cancel my booking?",
            use_nli=False
        )
        assert any("False universal guarantee" in claim for claim in result.unsupported_claims)
        assert result.grounding_score < 6.0


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
