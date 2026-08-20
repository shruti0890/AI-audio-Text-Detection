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

# Curvature baseline stats from SmolLM2-135M (mu0=-0.659736, sigma0=0.224686)
_DEFAULT_CURVATURE_MU0 = -0.659736
_DEFAULT_CURVATURE_SIGMA0 = 0.224686


def score_sentences(text: str, baseline_stats: dict | None = None) -> list[dict]:
    """
    Split text into sentences and score each sentence using a sliding context window
    (target sentence + 1 before + 1 after) with probability curvature.

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
                    "curvature_score": float | None,
                    "low_context": bool
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

    n_sents = len(sentences)
    if n_sents == 0:
        return []

    is_low_context = (n_sents <= 2)

    scored = []
    for i, sent in enumerate(sentences):
        words = sent.split()
        w_count = len(words)

        # Build sliding context window
        if is_low_context:
            context_text = sent
        else:
            if i == 0:
                window = sentences[0:2]
            elif i == n_sents - 1:
                window = sentences[n_sents - 2:n_sents]
            else:
                window = sentences[i - 1:i + 2]
            context_text = " ".join(window)

        context_word_count = len(context_text.split())

        if context_word_count < 6:
            # Curvature is unreliable on tiny spans under 6 words
            scored.append({
                "sentence": sent,
                "word_count": w_count,
                "curvature_raw": None,
                "curvature_score": None,
                "low_context": True,
            })
            continue

        try:
            raw_curv = get_curvature(context_text, min_tokens=8)
            if raw_curv is None or sigma0 <= 0:
                c_score = None
            else:
                z = (raw_curv - mu0) / sigma0
                c_score = float(norm.cdf(z) * 100.0)
                c_score = round(max(0.0, min(100.0, c_score)), 2)
        except Exception as e:
            logger.warning("Error scoring sentence window %r: %s", context_text[:30], e)
            raw_curv = None
            c_score = None

        scored.append({
            "sentence": sent,
            "word_count": w_count,
            "curvature_raw": round(raw_curv, 4) if raw_curv is not None else None,
            "curvature_score": c_score,
            "low_context": is_low_context,
        })

    return scored
