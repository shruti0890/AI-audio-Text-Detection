"""
text_forensics/signals/sentence_scorer.py

Correction 8: Sentence-Level Explainability Engine (QuillBot-Style).

Evaluates probability curvature on individual sentences to highlight which specific
sentences drive the AI detection verdict.

Uses probability curvature (Fast-DetectGPT) exclusively, as it is the only signal
statistically valid at sentence length (burstiness, cliche density, and entropy
require paragraph-length text).
"""

from __future__ import annotations

import json
import logging

import nltk
from scipy.stats import norm

from text_forensics.signals.curvature import get_curvature

logger = logging.getLogger(__name__)

# Ensure NLTK tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

# Curvature baseline stats from HC3 (mu0=-1.346745, sigma0=0.32602)
_DEFAULT_CURVATURE_MU0 = -1.346745
_DEFAULT_CURVATURE_SIGMA0 = 0.326020


def score_sentences(text: str, baseline_stats: dict | None = None) -> list[dict]:
    """
    Split text into sentences and score each sentence using probability curvature.

    Args:
        text: Input text string.
        baseline_stats: Optional dict from baseline_stats.json.

    Returns:
        list[dict]: List of sentence dicts:
            [
                {
                    "sentence": str,
                    "word_count": int,
                    "curvature_raw": float | None,
                    "curvature_score": float | None
                },
                ...
            ]
    """
    if not text or not text.strip():
        return []

    # Get curvature baseline params
    mu0 = _DEFAULT_CURVATURE_MU0
    sigma0 = _DEFAULT_CURVATURE_SIGMA0
    if baseline_stats and "curvature" in baseline_stats:
        mu0 = baseline_stats["curvature"].get("mu0", mu0)
        sigma0 = baseline_stats["curvature"].get("sigma0", sigma0)

    # Tokenize into sentences
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in text.split(".") if s.strip()]

    scored = []
    for sent in sentences:
        words = sent.split()
        w_count = len(words)

        if w_count < 6:
            # Curvature is unreliable on tiny spans under 6 words
            scored.append({
                "sentence": sent,
                "word_count": w_count,
                "curvature_raw": None,
                "curvature_score": None,
            })
            continue

        try:
            raw_curv = get_curvature(sent, min_tokens=8)
            if raw_curv is None or sigma0 <= 0:
                c_score = None
            else:
                z = (raw_curv - mu0) / sigma0
                c_score = float(norm.cdf(z) * 100.0)
                c_score = round(max(0.0, min(100.0, c_score)), 2)
        except Exception as e:
            logger.warning("Error scoring sentence %r: %e", sent[:30], e)
            raw_curv = None
            c_score = None

        scored.append({
            "sentence": sent,
            "word_count": w_count,
            "curvature_raw": round(raw_curv, 4) if raw_curv is not None else None,
            "curvature_score": c_score,
        })

    return scored
