"""
text_forensics/pipeline.py

Public Interface: Text Forensics Pipeline — Five-Feature Production Architecture.

This module is the single entry point for the Text Forensics Pipeline.
Other modules (cross-modal consistency checker, tagging engine, web UI) MUST
use only analyze_text() from this module.

Two paths run simultaneously:
  PATH A — Four-Feature Corrected Baseline (legacy fallback, preserved for backward compatibility)
  PATH B — Five-Feature Logistic Regression (PRODUCTION MODEL)

LOCKED PUBLIC INTERFACE:

    analyze_text(text: str, run_robustness: bool = True) -> dict

    Return schema:
    {
        # ── PATH A — Four-feature baseline (preserved) ─────────────────────
        "text_score": float,           # 0-100 fused AI-likelihood score (PATH A)
        "signal_agreement": str,       # "agreement" or "disagreement"
        "signals": {
            "curvature_raw": float | None,
            "curvature_score": float | None,
            "burstiness_raw": float | None,
            "burstiness_score": float | None,
            "cliche_density_pct": float,
            "cliche_score": float,
            "ttr": float,
            "entropy": float,
            "entropy_score": float,
        },
        "stability_flag": str,
        "paraphrase_delta": float,
        "compared_on_truncated": bool,

        # ── PATH B — Five-feature LR (PRODUCTION) ──────────────────────────
        "ai_probability": float,       # P(AI|X) in [0, 1] from 5-feature LR model
        "ai_score": float,             # ai_probability * 100 in [0, 100]
        "verdict": str,                # "Human" | "Likely Human" | "Likely AI" | "AI"
        "confidence": str,             # "Low" | "Moderate" | "High"
        "model_used": str,             # "five_feature_logistic_regression"

        "features": {                  # all 5 raw feature values
            "curvature": float | None,
            "burstiness": float | None,
            "lexical_entropy": float | None,
            "structural_regularity": float | None,
            "cliche_density": float,
        },

        "sentence_evidence": {         # contextual sentence-level AI summary
            "mean_ai_probability": float | None,
            "upper_quartile": float | None,
            "ai_sentence_ratio": float | None,
            "n_sentences_analyzed": int,
        },

        "short_text_warning": bool,    # True if text has < 30 words or < 5 sentences
    }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load calibration baseline stats at module import time
# ---------------------------------------------------------------------------
_BASELINE_STATS_PATH = Path(__file__).parent / "calibration" / "baseline_stats.json"
_BASELINE_STATS: Optional[dict] = None


def _load_baseline_stats() -> Optional[dict]:
    """
    Load baseline_stats.json from the calibration directory.

    Returns:
        dict: Calibration stats, or empty dict if missing.
    """
    global _BASELINE_STATS
    if _BASELINE_STATS is not None:
        return _BASELINE_STATS

    if not _BASELINE_STATS_PATH.exists():
        logger.warning(
            "baseline_stats.json not found at %s. "
            "Scores will use fallback (mu0=0, sigma0=1) defaults.",
            _BASELINE_STATS_PATH,
        )
        _BASELINE_STATS = {}
        return _BASELINE_STATS

    with open(_BASELINE_STATS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        _BASELINE_STATS = {}
        return _BASELINE_STATS

    _BASELINE_STATS = data
    logger.info("Loaded baseline stats from %s (sample_size=%s)", _BASELINE_STATS_PATH, data.get("sample_size", "?"))
    return _BASELINE_STATS


# Eagerly load at import time
_load_baseline_stats()


def _compute_sentence_evidence(
    sentences_scored: list[dict],
    ai_threshold: float = 0.50,
) -> dict:
    """
    Aggregate sentence-level AI probabilities into document-level contextual evidence.

    Args:
        sentences_scored: list of dicts with "curvature_score" (0-100) from sentence_scorer.
        ai_threshold: probability above which a sentence is considered AI-like.

    Returns:
        dict with mean_ai_probability, upper_quartile, ai_sentence_ratio, n_sentences_analyzed.
    """
    probs = []
    for s in sentences_scored:
        cs = s.get("curvature_score")
        if cs is not None:
            probs.append(cs / 100.0)

    if not probs:
        return {
            "mean_ai_probability": None,
            "upper_quartile": None,
            "ai_sentence_ratio": None,
            "n_sentences_analyzed": 0,
        }

    import statistics
    mean_prob = statistics.mean(probs)
    sorted_probs = sorted(probs)
    n = len(sorted_probs)
    q3_idx = int(0.75 * n)
    upper_q = sorted_probs[min(q3_idx, n - 1)]
    ai_sentence_ratio = sum(1 for p in probs if p >= ai_threshold) / n

    return {
        "mean_ai_probability": round(mean_prob, 4),
        "upper_quartile": round(upper_q, 4),
        "ai_sentence_ratio": round(ai_sentence_ratio, 4),
        "n_sentences_analyzed": n,
    }


def analyze_text(text: str, run_robustness: bool = True) -> dict:
    """
    Full text forensics pipeline — Five-Feature Production Architecture.

    Runs both:
      PATH A — four-feature corrected baseline (text_score)
      PATH B — five-feature Logistic Regression (ai_probability, verdict)

    Args:
        text: Non-empty string to analyze. Must be a str type.
        run_robustness: Whether to run the T5 paraphrase robustness check (default True).

    Returns:
        dict: Matching the locked production schema.

    Raises:
        TypeError: If text is not a str.
        ValueError: If text is empty or whitespace-only.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"analyze_text() expects a str, got {type(text).__name__!r}. "
            "Pass the text as a plain Python string."
        )
    if not text or not text.strip():
        raise ValueError(
            "analyze_text() received an empty or whitespace-only string. "
            "Provide actual text content to analyze."
        )

    clean_text = text.strip()
    words = clean_text.split()
    word_count = len(words)
    short_text_warning = word_count < 30

    # ---- Import signals lazily ----
    from text_forensics.signals.curvature import get_curvature
    from text_forensics.signals.burstiness import get_burstiness
    from text_forensics.signals.cliche_scanner import get_cliche_density
    from text_forensics.signals.lexical_entropy import get_lexical_stats
    from text_forensics.signals.structural_regularity import get_structural_regularity
    from text_forensics.fusion import compute_text_score, compute_five_feature_score
    from text_forensics.robustness_test import check_stability
    from text_forensics.signals.sentence_scorer import score_sentences

    baseline = _load_baseline_stats()

    # ========================================================================
    # FEATURE EXTRACTION — 5 core production features
    # ========================================================================
    logger.info("Running curvature signal...")
    curvature_raw = get_curvature(clean_text)

    logger.info("Running burstiness signal...")
    burstiness_raw = get_burstiness(clean_text)

    logger.info("Running cliche density signal...")
    cliche_pct = get_cliche_density(clean_text)

    logger.info("Running lexical entropy signal...")
    lex = get_lexical_stats(clean_text)
    entropy_val = lex["entropy"]
    ttr_val = lex["ttr"]

    logger.info("Running structural regularity signal...")
    struct_result = get_structural_regularity(clean_text)
    struct_composite = struct_result["composite"]

    # ========================================================================
    # PATH A — Four-feature corrected baseline (legacy fallback)
    # ========================================================================
    raw_signals = {
        "curvature_raw": curvature_raw,
        "burstiness_raw": burstiness_raw,
        "cliche_density_raw": cliche_pct,
        "entropy_raw": entropy_val,
    }

    fusion_result = compute_text_score(raw_signals, baseline)
    text_score = fusion_result["text_score"]
    sub_scores = fusion_result["sub_scores"]

    # ========================================================================
    # PATH B — Five-feature Logistic Regression (PRODUCTION)
    # ========================================================================
    lr_result = compute_five_feature_score(
        curvature=curvature_raw,
        burstiness=burstiness_raw,
        lexical_entropy=entropy_val,
        structural_regularity=struct_composite,
        cliche_density=cliche_pct,
        baseline_stats=baseline,
    )

    # ========================================================================
    # SENTENCE-LEVEL EVIDENCE
    # ========================================================================
    logger.info("Running sentence-level scoring...")
    try:
        sentence_analysis = score_sentences(clean_text, baseline)
        sentence_evidence = _compute_sentence_evidence(sentence_analysis)
    except Exception as e:
        logger.warning("Sentence scoring failed: %s", e)
        sentence_analysis = []
        sentence_evidence = {
            "mean_ai_probability": None,
            "upper_quartile": None,
            "ai_sentence_ratio": None,
            "n_sentences_analyzed": 0,
        }

    # ========================================================================
    # ROBUSTNESS SELF-TEST
    # ========================================================================
    if run_robustness:
        logger.info("Running robustness self-test...")
        robustness = check_stability(clean_text, text_score, baseline)
    else:
        logger.info("Skipping robustness self-test per request.")
        robustness = {
            "stability_flag": "skipped",
            "paraphrase_delta": 0.0,
            "compared_on_truncated": False,
        }

    # ========================================================================
    # ASSEMBLE OUTPUT SCHEMA
    # ========================================================================
    result = {
        # ── PATH A (preserved keys) ──────────────────────────────────────────
        "text_score": text_score,
        "signal_agreement": fusion_result.get("signal_agreement", "agreement"),
        "signals": {
            "curvature_raw": curvature_raw,
            "curvature_score": sub_scores.get("curvature_score"),
            "burstiness_raw": burstiness_raw,
            "burstiness_score": sub_scores.get("burstiness_score"),
            "cliche_density_pct": cliche_pct,
            "cliche_score": sub_scores.get("cliche_density_score"),
            "ttr": ttr_val,
            "entropy": entropy_val,
            "entropy_score": sub_scores.get("entropy_score"),
        },
        "stability_flag": robustness["stability_flag"],
        "paraphrase_delta": robustness["paraphrase_delta"],
        "compared_on_truncated": robustness.get("compared_on_truncated", False),

        # ── PATH B (production keys) ─────────────────────────────────────────
        "ai_probability": lr_result["ai_probability"],
        "ai_score": lr_result["ai_score"],
        "verdict": lr_result["verdict"],
        "confidence": lr_result["confidence"],
        "model_used": lr_result["model_used"],

        "features": {
            "curvature": curvature_raw,
            "burstiness": burstiness_raw,
            "lexical_entropy": entropy_val,
            "structural_regularity": struct_composite,
            "cliche_density": cliche_pct,
        },

        "sentence_evidence": sentence_evidence,
        "short_text_warning": short_text_warning,

        # Subcomponent details for explainability
        "structural_details": {
            "starter_diversity": struct_result["starter_diversity"],
            "pos_similarity": struct_result["pos_similarity"],
            "discourse_density": struct_result["discourse_density"],
        },
    }

    return result
