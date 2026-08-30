"""
rag_verifier.py - Stage 3B: RAG Grounding Verification
ControlPlane.ai (PS1 Architecture)

Features:
1. Canonical BM25 Retriever (RetEngine): Token-boundary BM25 ranking across the 20 Airbnb policy documents with metadata boosting.
2. Dual-Gate Factual Grounding Verifier (RAGVerifier):
   - Factual Assertion Detection (eliminates "no docs = free pass")
   - Exact Numeric, Currency, and Timeframe Validation (24h, 5d, 14d, 30d, 72h, 15d)
   - Hybrid NLI Entailment Layer with Refusal/Disclaimer protection
   - Computes Grounding Score G (0-10), Verification Confidence (0.0-1.0), and VerificationStatus
"""

import math
import re
import logging
from typing import List, Tuple, Dict, Any, Set, Optional
from models import (
    Stage3BResult,
    KnowledgeChunk,
    VerificationStatus,
    CRAGStatus,
    Config
)

logger = logging.getLogger(__name__)

# Standard English stopwords to prevent BM25 IDF dilution
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "you", "your", "can", "our", "all", "any", "how", "what",
    "when", "where", "which", "who", "why", "does", "have", "this", "or", "but",
    "get", "got", "do", "did", "if", "my", "me", "i", "we", "us", "they", "them", "some"
}

# Regex Gate 1: Comprehensive numeric, currency, percentage, and timeframe constraints
NUMERIC_ENTITY_PATTERN = re.compile(
    r"(?:\$\s*\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|days?|business\s+days?|hours?|hrs?|nights?|minutes?|mins?|months?|years?))\b",
    re.IGNORECASE
)

# Regex Gate 2: Policy Commitment & Action Verbs
POLICY_ACTION_PATTERN = re.compile(
    r"\b(refund|refunds|cancel|cancelled|cancelling|cancellation|reimburse|reimbursement|payout|payouts|disburse|disbursement|guarantee|guarantees|eligible|eligibility|fee waiver|penalty|aircover|deposit|coverage)\b",
    re.IGNORECASE
)

# Regex Gate 3: Negation and Refutation Context Patterns
NEGATION_CONTEXT_PATTERN = re.compile(
    r"\b(cannot|can't|not\s+(?:eligible|entitled|refundable|subject|permitted|possible)|"
    r"does\s+not\s+(?:qualify|apply|cover|provide|allow)|won't\s+be\s+refunded|"
    r"no\s+(?:\w+\s+)?refund|ineligible|non-refundable|neither|without|is\s+not\s+covered)\b",
    re.IGNORECASE
)


def _tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric tokens excluding stopwords."""
    raw_tokens = re.findall(r"\b[a-z0-9_$-]{2,}\b", text.lower())
    return [t for t in raw_tokens if t not in STOPWORDS]


def evaluate_factual_assertions(text: str) -> bool:
    """
    Returns True if the text makes empirical numeric assertions or contractual policy claims.
    Used when zero documents are retrieved to distinguish benign conversation from ungrounded assertions.
    """
    has_numeric = bool(NUMERIC_ENTITY_PATTERN.search(text))
    has_policy_action = bool(POLICY_ACTION_PATTERN.search(text))
    return has_numeric or has_policy_action


# ============================================================================
# 1. Canonical BM25 Retriever & CRAG Confidence Evaluator (RetEngine)
# ============================================================================

def retrieve_knowledge_chunks_scored(
    query_text: str,
    top_k: int = 3,
    kb: Optional[List[KnowledgeChunk]] = None,
    product_filter: Optional[str] = None,
    region_filter: Optional[str] = None
) -> List[Tuple[float, KnowledgeChunk]]:
    """
    Canonical BM25 Knowledge Base Retriever (RetEngine) returning ranked (score, chunk) tuples.
    """
    if kb is None:
        try:
            from db import get_all_knowledge_chunks
            kb = get_all_knowledge_chunks()
        except Exception:
            kb = []

    if not kb:
        return []

    q_tokens = _tokenize(query_text)
    if not q_tokens:
        return []

    N = len(kb)
    doc_tokens_list = [_tokenize(chunk.title + " " + chunk.content + " " + " ".join(chunk.keywords)) for chunk in kb]
    avgdl = sum(len(dt) for dt in doc_tokens_list) / max(1, N)

    # Document frequency per token
    df: Dict[str, int] = {}
    for dt in doc_tokens_list:
        unique_tokens = set(dt)
        for t in unique_tokens:
            df[t] = df.get(t, 0) + 1

    k1 = 1.5
    b = 0.75
    scored: List[Tuple[float, KnowledgeChunk]] = []

    for idx, chunk in enumerate(kb):
        if product_filter and chunk.product and chunk.product not in (product_filter, "all"):
            continue
        if region_filter and chunk.region and chunk.region not in (region_filter, "global"):
            continue

        dt = doc_tokens_list[idx]
        dl = len(dt)
        score = 0.0

        # Term frequency in document
        tf: Dict[str, int] = {}
        for t in dt:
            tf[t] = tf.get(t, 0) + 1

        for q in q_tokens:
            if q in tf:
                n_q = df.get(q, 1)
                idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
                f_qd = tf[q]
                numerator = f_qd * (k1 + 1)
                denominator = f_qd + k1 * (1 - b + b * (dl / max(1.0, avgdl)))
                score += idf * (numerator / denominator)

        # Metadata boost for exact query matches in product / category / region
        lower_q = query_text.lower()
        if chunk.product and chunk.product != "all" and chunk.product in lower_q:
            score *= 1.3
        if chunk.region and chunk.region != "global" and chunk.region in lower_q:
            score *= 1.4
        if chunk.category and chunk.category.lower() in lower_q:
            score *= 1.2

        # Direct title token match boost
        title_tokens = set(_tokenize(chunk.title))
        if set(q_tokens).intersection(title_tokens):
            score += 2.0

        if score >= 1.0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def retrieve_knowledge_chunks(
    query_text: str,
    top_k: int = 3,
    kb: Optional[List[KnowledgeChunk]] = None,
    product_filter: Optional[str] = None,
    region_filter: Optional[str] = None
) -> List[KnowledgeChunk]:
    """Retrieves top-k authoritative knowledge chunks via BM25."""
    scored = retrieve_knowledge_chunks_scored(
        query_text=query_text,
        top_k=top_k,
        kb=kb,
        product_filter=product_filter,
        region_filter=region_filter
    )
    return [c for _, c in scored]


def evaluate_crag_retrieval_confidence(
    query_text: str,
    retrieved_chunks: List[KnowledgeChunk],
    scored_list: Optional[List[Tuple[float, KnowledgeChunk]]] = None
) -> Tuple[float, CRAGStatus]:
    """
    Corrective RAG (CRAG) Retrieval Quality Evaluator.
    
    Computes a normalized confidence metric rho in [0.0, 1.0] from:
    1. Top BM25 score magnitude (normalized against benchmark ceiling).
    2. Query keyword token coverage across retrieved document titles and content.
    """
    if not retrieved_chunks:
        return 0.0, CRAGStatus.KNOWLEDGE_GAP_ABSTAIN

    top_score = scored_list[0][0] if (scored_list and len(scored_list) > 0) else 1.0
    norm_score = min(1.0, max(0.0, top_score / 6.0))

    q_tokens = set(_tokenize(query_text))
    if not q_tokens:
        coverage = 1.0
    else:
        top_doc_tokens = set(_tokenize(retrieved_chunks[0].title + " " + retrieved_chunks[0].content))
        overlap = len(q_tokens.intersection(top_doc_tokens))
        coverage = min(1.0, overlap / len(q_tokens))

    rho = round(0.30 * norm_score + 0.70 * coverage, 2)

    if rho >= Config.CRAG_HIGH_CONFIDENCE_THRESHOLD:
        status = CRAGStatus.HIGH_CONFIDENCE
    elif rho >= Config.CRAG_ABSTENTION_THRESHOLD:
        status = CRAGStatus.AMBIGUOUS
    else:
        status = CRAGStatus.KNOWLEDGE_GAP_ABSTAIN

    return rho, status


# ============================================================================
# 2. Numeric & Claim Grounding Verifier (RAGVerifier)
# ============================================================================

def _extract_numeric_entities(text: str) -> List[str]:
    """Extracts all numbers, currency, and timeframe spans from text, stripping trailing punctuation."""
    matches = NUMERIC_ENTITY_PATTERN.findall(text)
    cleaned = []
    for m in matches:
        token = re.sub(r"\s+", " ", m.strip()).rstrip(",.:;!?")
        if token:
            cleaned.append(token)
    return cleaned


def _is_number_in_negation_clause(num_str: str, text: str) -> bool:
    """
    Checks if a numeric entity is inside a negation/denial clause (e.g. 'does not qualify for $2,000').
    Prevents false-positive penalties when the bot refutes ungrounded user numbers.
    """
    num_escaped = re.escape(num_str)
    for m in re.finditer(num_escaped, text, re.IGNORECASE):
        start_idx = max(0, m.start() - 60)
        preceding = text[start_idx:m.start()]
        # Check sentence window
        sent_start = max(0, text.rfind(".", 0, m.start()) + 1)
        sent_end = min(len(text), m.end() + 40)
        sent_window = text[sent_start:sent_end]
        if NEGATION_CONTEXT_PATTERN.search(preceding) or NEGATION_CONTEXT_PATTERN.search(sent_window):
            return True
    return False


def _normalize_token(val: str) -> str:
    """Normalizes string for exact match (e.g. '$500' -> '$500', '30 days' -> '30 days')."""
    return re.sub(r"[,\.:;!?]+$", "", re.sub(r"\s+", " ", val.strip().lower()))


def _run_nli_entailment(
    sentence: str,
    source_text: str
) -> Tuple[bool, str]:
    """
    Hybrid NLI Entailment Layer: Uses LLM to verify whether a candidate claim
    is entailed, contradicted, or unsupported by the source text.
    """
    try:
        from llm_client import call_llm
        
        nli_prompt = (
            f"Premise (Source Policy):\n\"{source_text}\"\n\n"
            f"Hypothesis (Candidate Claim):\n\"{sentence}\"\n\n"
            "Does the premise entail, contradict, or leave unsupported the hypothesis?\n"
            "Return ONLY a JSON object: {\"label\": \"entailed\" | \"contradicted\" | \"unsupported\", "
            "\"reason\": \"<brief explanation>\"}"
        )
        
        res = call_llm(
            prompt=nli_prompt,
            system_instruction="You are a strict Natural Language Inference (NLI) evaluator.",
            json_mode=True
        )
        
        if isinstance(res, dict):
            label = str(res.get("label", "unsupported")).lower()
            return label == "entailed", label
            
    except Exception as e:
        logger.debug(f"NLI Entailment fallback to heuristic due to: {e}")

    # Fallback heuristic: token overlap
    s_tokens = set(re.findall(r"\b\w+\b", sentence.lower()))
    src_tokens = set(re.findall(r"\b\w+\b", source_text.lower()))
    overlap = len(s_tokens.intersection(src_tokens)) / max(len(s_tokens), 1)
    is_entailed = overlap >= 0.40
    return is_entailed, "entailed" if is_entailed else "unsupported"


def verify_factual_grounding(
    candidate_response: str,
    query: str = "",
    knowledge_base: Optional[List[KnowledgeChunk]] = None,
    use_nli: bool = True
) -> Stage3BResult:
    """
    Factual Grounding Verifier (Stage 3B) with Corrective RAG & Negation Awareness.
    
    1. Evaluates CRAG retrieval confidence rho in [0.0, 1.0].
    2. Active Abstention when rho < 0.40 on empirical policy claims.
    3. Negation-aware numeric entity validation.
    4. Sentence-level NLI entailment with disclaimer immunity.
    """
    # 1. Retrieve relevant policy chunks and evaluate query CRAG confidence
    query_target = query.strip() if query.strip() else candidate_response.strip()
    query_scored_chunks = retrieve_knowledge_chunks_scored(query_target, top_k=3, kb=knowledge_base)

    search_text = f"{query} {candidate_response}".strip()
    scored_chunks = retrieve_knowledge_chunks_scored(search_text, top_k=3, kb=knowledge_base)
    retrieved_chunks = [c for _, c in scored_chunks] if scored_chunks else [c for _, c in query_scored_chunks]
    
    crag_confidence, crag_status = evaluate_crag_retrieval_confidence(
        query_text=query_target,
        retrieved_chunks=retrieved_chunks,
        scored_list=query_scored_chunks
    )

    # 2. Case: Zero Knowledge Base Documents or CRAG Knowledge Gap (rho < 0.40)
    if crag_status == CRAGStatus.KNOWLEDGE_GAP_ABSTAIN or not retrieved_chunks:
        has_claims = evaluate_factual_assertions(candidate_response) or evaluate_factual_assertions(query)
        if has_claims:
            # Active Abstention: Unverified empirical or policy claims made with weak/no grounding
            return Stage3BResult(
                retrieved_chunks=retrieved_chunks,
                grounding_score=2.50,
                unsupported_claims=["CRAG Abstention: Policy assertion made with insufficient retrieval grounding."],
                numeric_mismatches=[],
                rag_risk=7.50,
                verification_confidence=0.0,
                verification_status=VerificationStatus.UNVERIFIED_ASSERTION,
                crag_confidence=crag_confidence,
                crag_status=CRAGStatus.KNOWLEDGE_GAP_ABSTAIN
            )
        else:
            # Benign conversation / pleasantry (e.g. "Hello, how can I help?")
            return Stage3BResult(
                retrieved_chunks=retrieved_chunks,
                grounding_score=10.0,
                unsupported_claims=[],
                numeric_mismatches=[],
                rag_risk=0.0,
                verification_confidence=1.0,
                verification_status=VerificationStatus.GENERAL_CONVERSATION,
                crag_confidence=1.0,
                crag_status=CRAGStatus.HIGH_CONFIDENCE
            )

    # 3. Case: Sufficient Grounding (rho >= 0.40) — Run Dual-Gate Verification
    combined_source_text = "\n\n".join([f"[{c.title}]: {c.content}" for c in retrieved_chunks])
    combined_source_lower = combined_source_text.lower()
    
    numeric_mismatches = []
    unsupported_claims = []
    penalty = 0.0

    # Gate A: Absolute Universal Guarantee & Over-Generalization Contradiction Check
    ABSOLUTE_GUARANTEE_PATTERN = re.compile(
        r"\b(every\s+\w+\s+(?:gives|has|qualifies|provides|offers|appears)|"
        r"unconditionally\s+guaranteed|guarantees?\s+(?:a\s+)?(?:full\s+)?refund|"
        r"regardless\s+of\s+(?:the\s+|your\s+)?(?:host|listing|standard|specific)?\s*(?:tier|policy)|"
        r"guarantees\s+that|automatically\s+(?:qualif\w+|activat\w+|pays|refund\w+|covers)|"
        r"overriding\s+any\s+(?:host|listing)\s+policy|retroactively\s+apply|"
        r"must\s+now\s+follow\s+the\s+newly\s+changed|automatically\s+pays\s+100%|"
        r"100%\s+full\s+refund\s+for\s+your\s+reservation)\b",
        re.IGNORECASE
    )
    if ABSOLUTE_GUARANTEE_PATTERN.search(candidate_response):
        unsupported_claims.append("False universal guarantee contradicts conditional policy terms.")
        penalty += 5.5

    # Gate B: Negation-Aware Numeric, Currency, & Timeframe Grounding
    candidate_numbers = _extract_numeric_entities(candidate_response)
    source_numbers = set([_normalize_token(n) for n in _extract_numeric_entities(combined_source_text)])

    for num in candidate_numbers:
        norm_num = _normalize_token(num)
        # Check exact token match or substring match in source
        if norm_num not in source_numbers and norm_num not in combined_source_lower:
            # Check if this number is in a denial/negation clause (e.g. 'does not qualify for $2,000')
            if _is_number_in_negation_clause(num, candidate_response):
                logger.debug(f"Numeric claim '{num}' ignored due to negation/refutation context.")
                continue
            numeric_mismatches.append(f"Unverified numeric claim '{num}' not found in source policy.")
            penalty += 3.5

    # Gate C: Sentence-Level Fact Consistency (NLI Entailment)
    sentences = [s.strip() for s in re.split(r"[.!?]", candidate_response) if len(s.strip()) > 15]
    source_tokens = set(re.findall(r"\b\w+\b", combined_source_lower))
    
    # Common meta-refusals and capability disclaimers that do not assert factual claims
    DISCLAIMER_PATTERNS = re.compile(
        r"^(i cannot|i am unable to|as an ai|i do not have|please contact|to find this|for more information)\b",
        re.IGNORECASE
    )

    for sentence in sentences:
        if DISCLAIMER_PATTERNS.search(sentence.strip()):
            continue

        s_tokens = set(re.findall(r"\b\w+\b", sentence.lower()))
        s_content = {t for t in s_tokens if len(t) > 3 and t not in STOPWORDS}
        
        if not s_content:
            continue

        overlap_ratio = len(s_content.intersection(source_tokens)) / max(len(s_content), 1)

        if overlap_ratio < 0.25:
            # Low token overlap — verify via semantic NLI entailment
            if use_nli:
                is_supported, nli_label = _run_nli_entailment(sentence, combined_source_text)
            else:
                is_supported, nli_label = False, "unsupported"
            if not is_supported:
                unsupported_claims.append(
                    f"Ungrounded claim (NLI: {nli_label}): '{sentence}'"
                )
                penalty += 3.5 if nli_label == "contradicted" else 2.5

    # 4. Compute Final Grounding Score, Confidence, and Status
    grounding_score = max(0.0, min(10.0, round(10.0 - penalty, 2)))
    rag_risk = round(10.0 - grounding_score, 2)

    if grounding_score >= 7.0:
        status = VerificationStatus.VERIFIED_GROUNDED
        confidence = 1.0
    elif grounding_score >= 3.0:
        status = VerificationStatus.PARTIALLY_GROUNDED
        confidence = round(grounding_score / 10.0, 2)
    else:
        status = VerificationStatus.CONTRADICTED
        confidence = 0.0

    return Stage3BResult(
        retrieved_chunks=retrieved_chunks,
        grounding_score=grounding_score,
        unsupported_claims=unsupported_claims,
        numeric_mismatches=numeric_mismatches,
        rag_risk=rag_risk,
        verification_confidence=confidence,
        verification_status=status,
        crag_confidence=crag_confidence,
        crag_status=crag_status
    )
