"""
signals/lexical_entropy.py

Signal 4: Lexical Diversity & Entropy.

Reference: Gehrmann et al. "GLTR: Statistical Detection and Visualization of Generated Text",
ACL 2019, arXiv:1906.04043.

AI-generated text tends to use a narrower vocabulary (lower TTR) and concentrate
probability mass on fewer high-frequency words (lower entropy) compared to human text.

Implementation notes:
- Uses spaCy en_core_web_sm for tokenization and alphabetic filtering.
- For texts longer than 400 words, TTR is computed via a rolling 200-word window and
  averaged, to correct for the well-known length bias of global TTR (Malvern et al. 2004).
  Shannon entropy is still computed globally (it is less length-sensitive than TTR).
- Only alphabetic tokens (token.is_alpha) are included; punctuation, numbers, and symbols
  are excluded from both TTR and entropy.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded spaCy model — loaded once at first call
_NLP = None

# Threshold above which rolling-window TTR is used instead of global TTR
_ROLLING_THRESHOLD = 400

# Window size for rolling TTR (Malvern et al. recommend 100-200 words)
_WINDOW_SIZE = 200


def _load_spacy():
    """Load spaCy en_core_web_sm model into module-level cache."""
    global _NLP
    if _NLP is None:
        import spacy
        logger.info("Loading spaCy en_core_web_sm model...")
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
        logger.info("spaCy model loaded.")
    return _NLP


def _rolling_ttr(tokens: list[str], window: int) -> float:
    """
    Compute the mean TTR over a rolling window of `window` words.

    This corrects for the length bias of global TTR: as text grows, global TTR
    always decreases even for diverse writing, because rare words accumulate slower
    than total tokens.

    Args:
        tokens: List of lowercase alphabetic word tokens.
        window: Window size in words.

    Returns:
        float: Mean TTR over all windows. If len(tokens) < window, returns global TTR.
    """
    if len(tokens) <= window:
        # Text shorter than window — fall back to global TTR
        return len(set(tokens)) / len(tokens) if tokens else 0.0

    ttrs = []
    for start in range(0, len(tokens) - window + 1, window):
        window_tokens = tokens[start : start + window]
        ttrs.append(len(set(window_tokens)) / len(window_tokens))
    return sum(ttrs) / len(ttrs) if ttrs else 0.0


def get_lexical_stats(text: str) -> dict:
    """
    Compute Type-Token Ratio (TTR) and Shannon entropy for the text's word distribution.

    For texts with more than 400 alphabetic words, TTR is computed using a rolling
    200-word window and averaged to correct for length bias (see module docstring).
    Entropy is always computed globally.

    Args:
        text: The input text. May be any length, including a single sentence.

    Returns:
        dict with keys:
          "ttr" (float): Type-Token Ratio in [0.0, 1.0]. Higher = more diverse vocabulary.
          "entropy" (float): Shannon entropy in bits. Higher = more uniform distribution.
        Returns {"ttr": 0.0, "entropy": 0.0} for empty or non-alphabetic text.
    """
    if not text or not text.strip():
        return {"ttr": 0.0, "entropy": 0.0}

    nlp = _load_spacy()

    # Tokenize and filter to alphabetic tokens only
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.is_alpha]

    if not tokens:
        logger.warning("get_lexical_stats: no alphabetic tokens found in text. Returning zeros.")
        return {"ttr": 0.0, "entropy": 0.0}

    total_words = len(tokens)

    # ---- TTR (with rolling window for long texts) ----
    if total_words > _ROLLING_THRESHOLD:
        ttr = _rolling_ttr(tokens, _WINDOW_SIZE)
    else:
        ttr = len(set(tokens)) / total_words

    # ---- Shannon Entropy ----
    freq = Counter(tokens)
    entropy = 0.0
    for count in freq.values():
        p_i = count / total_words
        entropy -= p_i * math.log2(p_i)

    return {"ttr": round(ttr, 6), "entropy": round(entropy, 6)}
