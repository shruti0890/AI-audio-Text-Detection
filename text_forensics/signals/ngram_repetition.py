"""
signals/ngram_repetition.py

Signal 4 (NEW): N-gram Repetition Score.

Motivation:
    AI-generated text can exhibit repeated local phrase patterns that curvature,
    entropy, and burstiness may miss. This feature measures how much of the
    text's bigram/trigram/4-gram vocabulary consists of repeated constructions.

Research basis:
    N-gram repetition as an AI-text signal is discussed in the AI-generated text
    detection literature including "MoSEs: Uncertainty-Aware AI-Generated Text
    Detection" (EMNLP 2025). The specific R2/R3/R4 averaging formula used here is
    our project-specific engineering choice, not copied verbatim from that paper.

Definition:
    For each n ∈ {2, 3, 4}:

        R_n = (number of unique n-grams occurring more than once)
              / (total number of unique n-grams)

    Composite:
        R_ngram = (R2 + R3 + R4) / 3

    The composite is then mapped to a 0–100 scale consistent with the other signals
    using a Gaussian CDF calibration step (done in the fusion layer), or returned
    as the raw [0, 1] ratio for the feature extractor to handle.

Implementation notes:
    - Tokenises on whitespace, lowercases, strips punctuation.
    - Graceful fallback for texts too short to yield n-grams for a given n.
    - Does NOT require spaCy; uses only the standard library.
    - Returns a dict with individual R2/R3/R4 values plus the composite, for
      auditability in the debug report.

Minimum token requirement:
    - R2 requires >= 2 tokens (trivially met by any real text)
    - R3 requires >= 3 tokens
    - R4 requires >= 4 tokens
    - If tokens < 4, R4 falls back to None; composite uses available Rn only.
    - If total tokens < 20 (very short text), all Rn are returned but flagged.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Minimum total tokens for a meaningful score (logged as a warning if not met)
_MIN_TOKENS = 20

# Minimum n-gram count needed to compute a meaningful ratio
_MIN_NGRAMS_FOR_RELIABLE = 5


def _tokenize(text: str) -> list[str]:
    """
    Tokenise text into lowercase alphabetic/numeric word tokens.

    Strips punctuation, lowercases, splits on whitespace. Does not use
    external NLP libraries so this function is very fast and dependency-free.

    Args:
        text: Input text string.

    Returns:
        list[str]: List of lowercase word tokens.
    """
    # Remove non-alphanumeric (keep spaces for splitting)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = cleaned.lower().split()
    return tokens


def _ngram_repetition_ratio(tokens: list[str], n: int) -> Optional[float]:
    """
    Compute the repetition ratio R_n for n-grams of size n.

        R_n = (# unique n-grams occurring > once) / (# unique n-grams)

    Returns None if there are fewer than n tokens (impossible to form any n-gram),
    or if fewer than _MIN_NGRAMS_FOR_RELIABLE unique n-grams exist (unreliable).

    Args:
        tokens: List of lowercase tokens.
        n: N-gram size (2, 3, or 4).

    Returns:
        float in [0.0, 1.0], or None if insufficient data.
    """
    if len(tokens) < n:
        return None

    # Build all n-grams as tuples
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)

    unique_ngrams = len(counts)
    if unique_ngrams < _MIN_NGRAMS_FOR_RELIABLE:
        # Too few unique n-grams for a reliable ratio
        return None

    repeated_unique = sum(1 for c in counts.values() if c > 1)
    return repeated_unique / unique_ngrams


def get_ngram_repetition(text: str) -> Dict[str, Optional[float]]:
    """
    Compute bigram (R2), trigram (R3), and 4-gram (R4) repetition ratios and
    their average composite R_ngram.

    The composite is:
        R_ngram = mean of available Rn values  (R2, R3, R4 each in [0.0, 1.0])

    If none of R2/R3/R4 can be computed (extreme edge case — text too short),
    returns all None. The composite is None only if all three are None.

    Args:
        text: Input text string. May be empty; returns all-None gracefully.

    Returns:
        dict with keys:
            "r2"        (float | None): bigram repetition ratio
            "r3"        (float | None): trigram repetition ratio
            "r4"        (float | None): 4-gram repetition ratio
            "composite" (float | None): mean of available Rn values in [0.0, 1.0]
            "token_count" (int):        number of tokens processed
            "short_text"  (bool):       True if < _MIN_TOKENS (score flagged)
    """
    fallback = {
        "r2": None,
        "r3": None,
        "r4": None,
        "composite": None,
        "token_count": 0,
        "short_text": True,
    }

    if not text or not text.strip():
        logger.warning("get_ngram_repetition: received empty text. Returning all-None.")
        return fallback

    tokens = _tokenize(text)
    token_count = len(tokens)

    if token_count < 4:
        logger.warning(
            "get_ngram_repetition: only %d tokens (< 4 minimum). "
            "Cannot compute any n-gram repetition ratios. Returning all-None.",
            token_count,
        )
        return {**fallback, "token_count": token_count}

    is_short = token_count < _MIN_TOKENS
    if is_short:
        logger.info(
            "get_ngram_repetition: %d tokens (< %d recommended). "
            "Scores computed but flagged as short_text=True.",
            token_count,
            _MIN_TOKENS,
        )

    r2 = _ngram_repetition_ratio(tokens, 2)
    r3 = _ngram_repetition_ratio(tokens, 3)
    r4 = _ngram_repetition_ratio(tokens, 4)

    available = [v for v in [r2, r3, r4] if v is not None]
    composite = round(sum(available) / len(available), 6) if available else None

    logger.debug(
        "get_ngram_repetition: tokens=%d  R2=%.4f  R3=%.4f  R4=%.4f  composite=%.4f",
        token_count,
        r2 if r2 is not None else float("nan"),
        r3 if r3 is not None else float("nan"),
        r4 if r4 is not None else float("nan"),
        composite if composite is not None else float("nan"),
    )

    return {
        "r2": round(r2, 6) if r2 is not None else None,
        "r3": round(r3, 6) if r3 is not None else None,
        "r4": round(r4, 6) if r4 is not None else None,
        "composite": composite,
        "token_count": token_count,
        "short_text": is_short,
    }
