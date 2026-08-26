"""
fast_checks.py - Stage 3A: Fast Parallel Checks
ControlPlane.ai (PS1 Architecture)

Executes sub-20ms parallel deterministic and statistical checks on the candidate response:
1. Heuristic Agent: Output PII, Leaked API Keys & Banned Lexicon scanning.
2. Statistical Scorer: N-Gram Repetition / Degeneracy, Entropy (Perplexity proxy), and Semantic Proximity.
3. Parallel Execution Bus: Scatter-gather executor running checks concurrently.
"""

import re
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

from models import Stage3AResult, PIIEntity, Config
from pii import PII_PATTERNS, _has_interval_overlap


# Prioritized order so API keys and SSNs are matched before generic phone numbers
PRIORITIZED_PII_KEYS = ["API_KEY", "CREDIT_CARD", "SSN", "EMAIL", "PHONE"]

STOPWORDS = {"what", "is", "our", "the", "a", "an", "for", "to", "in", "of", "and", "can", "be", "your", "my", "we", "me"}


# ============================================================================
# 1. Heuristic Agent: Output PII & Banned Lexicon
# ============================================================================

def check_output_heuristics(candidate_response: str) -> Tuple[List[PIIEntity], List[str], float]:
    """
    Scans candidate LLM output for:
    1. Leaked sensitive PII (API keys, SSN, credit cards, phones, emails).
    2. Enterprise prohibited / banned lexicon terms.
    
    Returns:
        (detected_pii, banned_terms_hit, heuristic_risk_score [0-10])
    """
    detected_pii: List[PIIEntity] = []
    
    # 1. Output PII & Key Detection in prioritized order
    for entity_type in PRIORITIZED_PII_KEYS:
        pattern = PII_PATTERNS[entity_type]
        for match in pattern.finditer(candidate_response):
            start, end = match.start(), match.end()
            raw_text = match.group()
            
            if _has_interval_overlap(start, end, detected_pii):
                continue
                
            detected_pii.append(
                PIIEntity(
                    entity_type=entity_type,
                    text=raw_text,
                    start=start,
                    end=end
                )
            )

    # 2. Banned Lexicon Matching
    candidate_lower = candidate_response.lower()
    banned_hits: List[str] = []
    for term in Config.BANNED_LEXICON:
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, candidate_lower):
            banned_hits.append(term)

    # 3. Calculate Heuristic Risk Score (0.0 to 10.0 scale)
    heuristic_risk = 0.0
    
    for entity in detected_pii:
        if entity.entity_type in ["API_KEY", "SSN", "CREDIT_CARD"]:
            heuristic_risk = max(heuristic_risk, 9.5)
        else:
            heuristic_risk = max(heuristic_risk, 5.0)

    if banned_hits:
        heuristic_risk = max(heuristic_risk, 8.5 + min(1.5, len(banned_hits) * 0.5))

    return detected_pii, banned_hits, min(10.0, heuristic_risk)


# ============================================================================
# 2. Statistical & Info-Theoretic Scorer
# ============================================================================

def _tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric words."""
    return re.findall(r"\b\w+\b", text.lower())


def _compute_ngram_repetition(tokens: List[str], n: int = 3) -> float:
    """
    Calculates n-gram repetition ratio (detects degenerate LLM repetition loops).
    Ratio = 1.0 - (unique_ngrams / total_ngrams). High ratio indicates repetitive looping.
    """
    if len(tokens) < n:
        return 0.0
    
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
        
    unique_count = len(set(ngrams))
    repetition_ratio = 1.0 - (unique_count / len(ngrams))
    return round(repetition_ratio, 4)


def _compute_shannon_entropy(text: str) -> float:
    """
    Calculates Shannon Entropy (in bits) over character distribution.
    Serves as an info-theoretic proxy for perplexity / randomness.
    Standard English text is typically between 3.5 and 4.8 bits.
    """
    if not text:
        return 0.0
        
    counts = Counter(text)
    total = len(text)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return round(entropy, 4)


def _compute_semantic_overlap(prompt_tokens: List[str], resp_tokens: List[str]) -> float:
    """
    Calculates content-word overlap with prefix matching (ignoring stopwords).
    Handles morphological variants like return/returned, refund/refunds.
    """
    p_content = [t for t in prompt_tokens if t not in STOPWORDS and len(t) > 2]
    r_content = [t for t in resp_tokens if t not in STOPWORDS and len(t) > 2]
    
    if not p_content or not r_content:
        return 1.0  # Cannot determine mismatch on trivial prompts
        
    matches = 0
    for p in p_content:
        # Prefix match of length >= 4 (e.g. "retu" in "return" and "returned")
        prefix = p[:4] if len(p) >= 4 else p
        if any(r.startswith(prefix) for r in r_content):
            matches += 1
            
    return round(matches / len(p_content), 4)


def compute_statistical_scores(candidate_response: str, prompt: str) -> Dict[str, float]:
    """
    Computes statistical and info-theoretic anomaly metrics on candidate response.
    
    Returns:
        Dict with ngram_repetition, entropy (perplexity proxy), similarity, and stat_risk (0-10).
    """
    resp_tokens = _tokenize(candidate_response)
    prompt_tokens = _tokenize(prompt)

    # 1. N-Gram Repetition (Tri-gram loop detection)
    ngram_rep = _compute_ngram_repetition(resp_tokens, n=3)

    # 2. Shannon Entropy (Perplexity proxy)
    entropy = _compute_shannon_entropy(candidate_response)

    # 3. Prompt-Response Content Overlap
    overlap = _compute_semantic_overlap(prompt_tokens, resp_tokens)

    # 4. Statistical Anomaly Risk Calculation (0.0 to 10.0 scale)
    stat_risk = 0.0

    # Severe penalty for degenerate repetition loops (> 35% repetition)
    if ngram_rep > 0.35:
        stat_risk += (ngram_rep - 0.35) * 12.0

    # Anomaly if entropy is abnormally low (< 2.5 bits in long texts -> repetitive gibberish)
    if len(candidate_response) > 50 and entropy < 2.5:
        stat_risk += (2.5 - entropy) * 3.0

    # Anomaly if completely detached from prompt content words
    if len(resp_tokens) > 5 and overlap == 0.0:
        stat_risk += 2.0

    stat_risk = min(10.0, round(stat_risk, 2))

    return {
        "ngram_repetition": ngram_rep,
        "perplexity_proxy": entropy,
        "cosine_similarity": overlap,
        "stat_risk": stat_risk
    }


# ============================================================================
# 3. Parallel Execution Bus (Scatter-Gather)
# ============================================================================

def run_stage3a_fast_checks(candidate_response: str, prompt: str) -> Stage3AResult:
    """
    Stage 3A: Parallel Fast Check Bus
    
    Executes Heuristic Agent and Statistical Scorer concurrently in parallel threads.
    
    Args:
        candidate_response: The draft output generated by Primary LLM.
        prompt: The user input prompt.
        
    Returns:
        Stage3AResult aggregating heuristic and statistical findings.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        heuristic_future = executor.submit(check_output_heuristics, candidate_response)
        statistical_future = executor.submit(compute_statistical_scores, candidate_response, prompt)

        detected_pii, banned_hits, heuristic_risk = heuristic_future.result()
        stat_metrics = statistical_future.result()

    return Stage3AResult(
        output_pii=detected_pii,
        banned_lexicon_hits=banned_hits,
        heuristic_risk=heuristic_risk,
        perplexity_score=stat_metrics["perplexity_proxy"],
        ngram_repetition=stat_metrics["ngram_repetition"],
        cosine_similarity=stat_metrics["cosine_similarity"],
        stat_risk=stat_metrics["stat_risk"]
    )
