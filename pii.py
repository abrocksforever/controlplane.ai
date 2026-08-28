"""
pii.py - Stage 1: Pre-Execution Guardrails (Robust PII Redaction & Weighted Injection Guard)
ControlPlane.ai (PS1 Architecture)

Features:
1. Exact Interval-Based Non-Overlapping PII Detection: Prevents regex collision.
2. Reverse Offset Slicing Redaction: Replaces text by character index to prevent string offset corruption.
3. Severity-Weighted Injection Scoring: Distinguishes high-risk jailbreaks from mild probes.
"""

import re
import logging
from typing import List, Tuple, Optional
from models import Stage1Result, PIIEntity

logger = logging.getLogger(__name__)


# ============================================================================
# 1. PII Patterns (ReDoS-Safe, Bounded Lookups)
# ============================================================================

PII_PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "PHONE": re.compile(r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "API_KEY": re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16})\b")
}


def _passes_luhn_check(card_number: str) -> bool:
    """
    Validates a credit card number using the Luhn algorithm (ISO/IEC 7812-1).
    Filters out false positives from 16-digit order IDs, tracking numbers, etc.
    """
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    
    # Luhn: double every second digit from right, subtract 9 if > 9, sum all
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0


# ============================================================================
# 2. Weighted Adversarial Prompt Injection Signatures
# ============================================================================

# (Regex Pattern, Severity Weight from 0.0 to 10.0, Description)
WEIGHTED_INJECTION_SIGNATURES: List[Tuple[re.Pattern, float, str]] = [
    # Critical / High Severity Attacks (Immediate Block: Weight >= 8.0)
    (re.compile(r"you\s+are\s+now\s+(in\s+)?(developer\s+mode|dan|jailbroken)", re.IGNORECASE), 9.5, "DAN / Developer Mode Jailbreak"),
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.IGNORECASE), 9.0, "Direct Instruction Override"),
    (re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+(directives|rules|guidelines)", re.IGNORECASE), 9.0, "System Directive Disregard"),
    (re.compile(r"do\s+anything\s+now", re.IGNORECASE), 8.5, "DAN Persona Exploit"),
    (re.compile(r"bypass\s+(content\s+)?filter", re.IGNORECASE), 8.5, "Filter Bypass Command"),
    
    # Medium Severity Attacks (Risk Flagging: Weight 5.0 - 7.5)
    (re.compile(r"(reveal|print|dump|show|output)\s+(your\s+)?(system\s+prompt|initial\s+instructions|hidden\s+rules)", re.IGNORECASE), 7.0, "System Prompt Extraction Probe"),
    (re.compile(r"(what\s+are\s+your\s+hidden\s+rules|repeat\s+the\s+words\s+above)", re.IGNORECASE), 5.5, "Prompt Exfiltration Probe"),
]


# ============================================================================
# 3. Helper Functions: Interval Overlap & Reverse Slice Redaction
# ============================================================================

def _has_interval_overlap(start: int, end: int, entities: List[PIIEntity]) -> bool:
    """Checks if character interval [start, end) intersects with any existing entity."""
    return any(not (end <= e.start or start >= e.end) for e in entities)


def _redact_by_reverse_offset(text: str, entities: List[PIIEntity]) -> str:
    """
    Redacts text using character slices sorted in descending order of start position.
    This guarantees that slice modifications at the end of the string never corrupt 
    the character offset positions of earlier entities.
    """
    # Sort descending by start index
    sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
    
    result = text
    for entity in sorted_entities:
        mask = f"[REDACTED_{entity.entity_type}]"
        result = result[:entity.start] + mask + result[entity.end:]
        
    return result


# ============================================================================
# 4. Main Stage 1 Function
# ============================================================================

def filter_input_pii_and_injection(prompt: str) -> Stage1Result:
    """
    Stage 1: Pre-Execution Guardrails
    
    1. Detects PII spans with strict character interval collision resolution.
    2. Performs reverse-offset slice redaction to prevent string corruption.
    3. Evaluates weighted prompt injection signatures.
    
    Args:
        prompt: Raw user input text.
        
    Returns:
        Stage1Result containing sanitized prompt, detected entities with exact offsets,
        weighted injection score, and early block flag if critical attack.
    """
    detected_pii: List[PIIEntity] = []

    # 1. Interval-Based PII Detection
    for entity_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(prompt):
            start, end = match.start(), match.end()
            raw_text = match.group()

            # Discard if character range overlaps with an already extracted entity
            if _has_interval_overlap(start, end, detected_pii):
                continue

            # Post-match Luhn validation for credit cards to filter false positives
            if entity_type == "CREDIT_CARD" and not _passes_luhn_check(raw_text):
                logger.debug(f"Credit card candidate '{raw_text}' failed Luhn check, skipping.")
                continue

            detected_pii.append(
                PIIEntity(
                    entity_type=entity_type,
                    text=raw_text,
                    start=start,
                    end=end
                )
            )

    # 2. Reverse-Offset Slicing Redaction
    sanitized_prompt = _redact_by_reverse_offset(prompt, detected_pii)

    # 3. Weighted Adversarial Prompt Injection Scoring
    matched_reasons = []
    max_severity = 0.0

    for pattern, weight, description in WEIGHTED_INJECTION_SIGNATURES:
        # Check both raw prompt and sanitized prompt
        if pattern.search(prompt) or pattern.search(sanitized_prompt):
            matched_reasons.append(f"{description} (Severity: {weight})")
            if weight > max_severity:
                max_severity = weight

    # Injection scoring and decision threshold
    if not matched_reasons:
        injection_score = 0.0
        is_injection = False
        is_blocked = False
        block_reason = None
    else:
        injection_score = min(10.0, max_severity)
        is_injection = True
        # Block if severity >= 8.0 (Critical attacks like DAN / Instruction Override)
        is_blocked = (injection_score >= 8.0)
        block_reason = " | ".join(matched_reasons)

    return Stage1Result(
        sanitized_prompt=sanitized_prompt,
        pii_detected=detected_pii,
        is_injection=is_injection,
        injection_score=injection_score,
        is_blocked=is_blocked,
        block_reason=block_reason
    )
