"""
text_forensics/pipeline.py

Public Interface: Text Forensics Pipeline — Full Implementation (Phase 1.9).

This module is the single entry point for the Text Forensics Pipeline.
Other modules (cross-modal consistency checker, tagging engine) MUST use
only analyze_text() from this module. The schema below is the locked contract.

LOCKED PUBLIC INTERFACE — do not change after Phase 1.9:

    analyze_text(text: str) -> dict

    Return schema:
    {
        "text_score": float,           # 0-100 fused AI-likelihood score (higher = more AI-like)
        "signals": {
            "curvature_raw": float | None,    # Fast-DetectGPT discrepancy (None if < 20 tokens)
            "curvature_score": float | None,  # calibrated 0-100 sub-score
            "burstiness_raw": float | None,   # σ/μ sentence-length ratio (None if < 5 sentences)
            "burstiness_score": float | None, # calibrated 0-100 sub-score
            "cliche_density_pct": float,      # cliché density as % of total words
            "cliche_score": float,            # calibrated 0-100 sub-score
            "ttr": float,                     # type-token ratio (rolling 200-word window if > 400 words)
            "entropy": float,                 # Shannon entropy in bits
            "entropy_score": float,           # calibrated 0-100 sub-score
        },
        "stability_flag": str,          # "stable", "unstable", or "unknown"
        "paraphrase_delta": float       # absolute score shift after paraphrasing
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
# Load calibration baseline stats at module import time (not on every call)
# ---------------------------------------------------------------------------
_BASELINE_STATS_PATH = Path(__file__).parent / "calibration" / "baseline_stats.json"
_BASELINE_STATS: Optional[dict] = None


def _load_baseline_stats() -> Optional[dict]:
    """
    Load baseline_stats.json from the calibration directory.

    Returns:
        dict: Calibration stats, or None if the file is missing/empty.
    """
    global _BASELINE_STATS
    if _BASELINE_STATS is not None:
        return _BASELINE_STATS

    if not _BASELINE_STATS_PATH.exists():
        logger.warning(
            "baseline_stats.json not found at %s. "
            "Run calibration/run_calibration.py first. "
            "Scores will use fallback (mu0=0, sigma0=1) defaults.",
            _BASELINE_STATS_PATH,
        )
        _BASELINE_STATS = {}
        return _BASELINE_STATS

    with open(_BASELINE_STATS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        logger.warning(
            "baseline_stats.json exists but is empty. "
            "Scores will use fallback (mu0=0, sigma0=1) defaults."
        )
        _BASELINE_STATS = {}
        return _BASELINE_STATS

    _BASELINE_STATS = data
    logger.info("Loaded baseline stats from %s (sample_size=%s)", _BASELINE_STATS_PATH, data.get("sample_size", "?"))
    return _BASELINE_STATS


# Eagerly load at import time
_load_baseline_stats()


def analyze_text(text: str, run_robustness: bool = True) -> dict:
    """
    Full text forensics pipeline.

    Validates input, loads calibration stats, runs all 4 signals (curvature,
    burstiness, cliche_density, lexical_entropy), fuses them into a calibrated
    0-100 AI-likelihood score, optionally runs the adversarial robustness self-test,
    and returns the complete result matching the locked schema.

    Args:
        text: Non-empty string to analyze. Must be a str type.
        run_robustness: Whether to run the T5 paraphrase robustness check (default True).

    Returns:
        dict: Matching the locked schema defined in this module's docstring.

    Raises:
        TypeError: If text is not a str.
        ValueError: If text is empty or whitespace-only.
    """
    # ---- Input validation ----
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

    # ---- Import signals (lazy — avoids loading models if pipeline not used) ----
    from text_forensics.signals.curvature import get_curvature
    from text_forensics.signals.burstiness import get_burstiness
    from text_forensics.signals.cliche_scanner import get_cliche_density
    from text_forensics.signals.lexical_entropy import get_lexical_stats
    from text_forensics.fusion import compute_text_score
    from text_forensics.robustness_test import check_stability

    baseline = _load_baseline_stats()

    # ---- Run all signals ----
    logger.info("Running curvature signal...")
    curvature_raw = get_curvature(text)

    logger.info("Running burstiness signal...")
    burstiness_raw = get_burstiness(text)

    logger.info("Running cliche density signal...")
    cliche_pct = get_cliche_density(text)

    logger.info("Running lexical entropy signal...")
    lex = get_lexical_stats(text)
    entropy_val = lex["entropy"]
    ttr_val = lex["ttr"]

    # ---- Fuse signals ----
    raw_signals = {
        "curvature_raw": curvature_raw,
        "burstiness_raw": burstiness_raw,
        "cliche_density_raw": cliche_pct,
        "entropy_raw": entropy_val,
    }

    fusion_result = compute_text_score(raw_signals, baseline)
    text_score = fusion_result["text_score"]
    sub_scores = fusion_result["sub_scores"]

    # ---- Robustness self-test ----
    if run_robustness:
        logger.info("Running robustness self-test...")
        robustness = check_stability(text, text_score, baseline)
    else:
        logger.info("Skipping robustness self-test per request.")
        robustness = {
            "stability_flag": "skipped",
            "paraphrase_delta": 0.0,
            "compared_on_truncated": False,
        }

    # ---- Assemble locked output schema ----
    result = {
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
        # Correction 4: expose whether the robustness delta used a truncated comparison
        "compared_on_truncated": robustness.get("compared_on_truncated", False),
    }

    return result
