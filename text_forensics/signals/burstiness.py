"""
signals/burstiness.py

Signal 2: Sentence-Length Burstiness.

Reference: Jawahar et al. 2020, "Automatic Detection of Machine Generated Text:
A Critical Survey", arXiv:2011.01314.

AI-generated text tends to have more uniform sentence lengths (low variance),
while human writing shows more rhythmic variation (high variance). The Fano-style
ratio σ/μ captures this: low B → uniform (AI-like), high B → varied (human-like).
"""

from __future__ import annotations

import logging
from typing import Optional

import nltk
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

# Minimum number of sentences required for a meaningful burstiness estimate
_MIN_SENTENCES = 5


def _ensure_nltk_punkt() -> None:
    """Download NLTK punkt tokenizer data if not already present."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("Downloading NLTK punkt_tab tokenizer data...")
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def get_burstiness(text: str) -> Optional[float]:
    """
    Compute the sentence-length burstiness score for the given text.

    Burstiness B = σ / μ where:
      - μ = mean word count per sentence
      - σ = standard deviation of word count per sentence

    A high B (> ~0.6) is consistent with human writing (variable rhythm).
    A low B (< ~0.3) is consistent with AI text (uniform sentence length).

    Guard clause: returns None if fewer than _MIN_SENTENCES (5) sentences
    are detected, because B is statistically unreliable on very short passages.

    Args:
        text: The input text to analyze. May be any length.

    Returns:
        float: Burstiness ratio B = σ/μ. Always >= 0.
        None: If fewer than _MIN_SENTENCES sentences are found.
    """
    if not text or not text.strip():
        logger.warning("get_burstiness: received empty or whitespace-only text. Returning None.")
        return None

    _ensure_nltk_punkt()

    sentences = sent_tokenize(text)
    n_sentences = len(sentences)

    if n_sentences < _MIN_SENTENCES:
        logger.warning(
            "get_burstiness: only %d sentences detected (< %d minimum). "
            "Returning None — burstiness would be unreliable.",
            n_sentences,
            _MIN_SENTENCES,
        )
        return None

    # Word count per sentence (split on whitespace — fast, sufficient for this metric)
    lengths = [len(s.split()) for s in sentences]

    n = len(lengths)
    mu = sum(lengths) / n

    if mu == 0.0:
        logger.warning("get_burstiness: mean sentence length is zero. Returning None.")
        return None

    variance = sum((l - mu) ** 2 for l in lengths) / n
    sigma = variance ** 0.5

    burstiness = sigma / mu
    return burstiness
