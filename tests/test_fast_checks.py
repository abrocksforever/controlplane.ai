"""
test_fast_checks.py - Tests for Stage 3A: Fast Parallel Checks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fast_checks import (
    check_output_heuristics,
    compute_statistical_scores,
    run_stage3a_fast_checks,
    _tokenize,
    _compute_ngram_repetition,
    _compute_shannon_entropy,
    _compute_semantic_overlap,
)


# ============================================================================
# Heuristic Agent Tests
# ============================================================================

class TestOutputHeuristics:
    """Tests for output PII detection and banned lexicon scanning."""

    def test_api_key_detected_in_output(self):
        response = "Your API key is sk-abcdefghijklmnopqrstuvwxyz123"
        pii, banned, risk = check_output_heuristics(response)
        assert len(pii) >= 1
        assert any(e.entity_type == "API_KEY" for e in pii)
        assert risk >= 9.0

    def test_ssn_detected_in_output(self):
        response = "The SSN on file is 123-45-6789."
        pii, _, risk = check_output_heuristics(response)
        assert any(e.entity_type == "SSN" for e in pii)
        assert risk >= 9.0

    def test_banned_lexicon_detected(self):
        response = "To exploit the system, bypass safety mechanisms."
        _, banned, risk = check_output_heuristics(response)
        assert "exploit" in banned or "bypass safety" in banned
        assert risk >= 8.5

    def test_clean_output_no_findings(self):
        response = "Your return window is 30 days from purchase."
        pii, banned, risk = check_output_heuristics(response)
        assert len(pii) == 0
        assert len(banned) == 0
        assert risk == 0.0


# ============================================================================
# Statistical Scorer Tests
# ============================================================================

class TestStatisticalScorer:
    """Tests for n-gram repetition, entropy, and semantic overlap."""

    def test_high_repetition_detected(self):
        """Degenerate looping text should have high repetition score."""
        looping_text = "buy now buy now buy now buy now buy now buy now buy now buy now"
        scores = compute_statistical_scores(looping_text, "product info")
        assert scores["ngram_repetition"] > 0.35
        assert scores["stat_risk"] > 0.0

    def test_normal_text_low_repetition(self):
        normal_text = "Standard items can be returned within 30 days of purchase for a full refund."
        scores = compute_statistical_scores(normal_text, "return policy")
        assert scores["ngram_repetition"] < 0.35

    def test_entropy_normal_range(self):
        """Normal English text should have entropy between 3.5 and 5.0 bits."""
        text = "The enterprise return policy allows customers to return items within thirty days."
        entropy = _compute_shannon_entropy(text)
        assert 3.0 <= entropy <= 5.5

    def test_low_entropy_on_repetitive_text(self):
        text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        entropy = _compute_shannon_entropy(text)
        assert entropy < 1.0

    def test_empty_text_zero_entropy(self):
        assert _compute_shannon_entropy("") == 0.0

    def test_semantic_overlap_high_for_related(self):
        prompt_tokens = _tokenize("return policy refund window")
        resp_tokens = _tokenize("returns are processed within the refund window period")
        overlap = _compute_semantic_overlap(prompt_tokens, resp_tokens)
        assert overlap > 0.5

    def test_semantic_overlap_zero_for_unrelated(self):
        prompt_tokens = _tokenize("quantum computing algorithms")
        resp_tokens = _tokenize("chocolate cake recipe ingredients baking")
        overlap = _compute_semantic_overlap(prompt_tokens, resp_tokens)
        assert overlap == 0.0


# ============================================================================
# N-Gram Repetition Tests
# ============================================================================

class TestNgramRepetition:
    """Tests for tri-gram loop detection."""

    def test_no_repetition(self):
        tokens = _tokenize("each word in this sentence is unique and different from others")
        rep = _compute_ngram_repetition(tokens, n=3)
        assert rep < 0.2

    def test_full_repetition(self):
        tokens = _tokenize("a b c a b c a b c a b c")
        rep = _compute_ngram_repetition(tokens, n=3)
        assert rep > 0.5

    def test_short_text_returns_zero(self):
        tokens = _tokenize("hi")
        rep = _compute_ngram_repetition(tokens, n=3)
        assert rep == 0.0


# ============================================================================
# Parallel Execution Tests
# ============================================================================

class TestParallelExecution:
    """Tests for Stage 3A scatter-gather bus."""

    def test_stage3a_returns_complete_result(self):
        result = run_stage3a_fast_checks(
            "Your refund will be processed within 5-7 business days.",
            "What is the refund timeline?"
        )
        assert hasattr(result, "heuristic_risk")
        assert hasattr(result, "stat_risk")
        assert hasattr(result, "output_pii")
        assert hasattr(result, "banned_lexicon_hits")
        assert hasattr(result, "ngram_repetition")
        assert hasattr(result, "perplexity_score")
        assert hasattr(result, "cosine_similarity")

    def test_clean_response_all_clear(self):
        result = run_stage3a_fast_checks(
            "Standard items can be returned within 30 days.",
            "return policy"
        )
        assert result.heuristic_risk == 0.0
        assert len(result.output_pii) == 0
        assert len(result.banned_lexicon_hits) == 0
