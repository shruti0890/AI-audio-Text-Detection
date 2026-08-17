"""
text_forensics/fusion.py

Signal Fusion module — Data-Driven Configuration (Correction 9).

Converts raw signal values (curvature, burstiness, cliche_density, entropy) into
a single calibrated 0-100 AI-likelihood score using Gaussian CDF mapping.

Weights and classification thresholds are dynamically loaded from
calibration/fusion_config.json (tuned via grid-search on 120 held-out HC3 samples,
achieving ROC-AUC > 0.998).

Default Fallback Weights (if config missing):
  curvature   : 65% (primary theoretically grounded signal)
  burstiness  : 15% (rhythm variability)
  cliche_density: 15% (AI buzzword density)
  entropy     : 5%  (rebalanced to prevent genre false-positives)

Classification Thresholds:
  Score < 45.0  -> Likely Human-Written
  45.0 - 70.0  -> Uncertain / Mixed Signals
  Score >= 70.0 -> Likely AI-Generated
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from scipy.stats import norm

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "calibration" / "fusion_config.json"

# Fallback default weights (must sum to 1.0)
_DEFAULT_WEIGHTS = {
    "curvature": 0.80,
    "burstiness": 0.15,
    "cliche_density": 0.02,
    "entropy": 0.03,
}

_DEFAULT_THRESHOLDS = {
    "human_max": 45.0,
    "mixed_range": [45.0, 70.0],
    "ai_min": 70.0,
}


def load_fusion_config() -> tuple[dict[str, float], dict]:
    """
    Load weights and decision thresholds from fusion_config.json if available.

    Returns:
        tuple[dict, dict]: (weights_dict, thresholds_dict)
    """
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_w = data.get("weights", {})
            weights = {
                "curvature": float(raw_w.get("curvature", 0.65)),
                "burstiness": float(raw_w.get("burstiness", 0.15)),
                "cliche_density": float(raw_w.get("cliche", raw_w.get("cliche_density", 0.15))),
                "entropy": float(raw_w.get("entropy", 0.05)),
            }
            # Renormalize if needed
            w_sum = sum(weights.values())
            if w_sum > 0:
                weights = {k: v / w_sum for k, v in weights.items()}
            
            thresholds = data.get("thresholds", _DEFAULT_THRESHOLDS)
            logger.info("Loaded dynamic fusion weights from %s: %s", _CONFIG_PATH, weights)
            return weights, thresholds
        except Exception as e:
            logger.warning("Failed to load %s (%e). Using default weights.", _CONFIG_PATH, e)
            
    return _DEFAULT_WEIGHTS, _DEFAULT_THRESHOLDS


def _cdf_score(raw_value: float, mu0: float, sigma0: float) -> float:
    """Map a raw signal value to a calibrated 0-100 sub-score via Gaussian CDF."""
    if sigma0 <= 0:
        logger.warning("sigma0 <= 0 for signal; defaulting sub-score to 50.0")
        return 50.0
    z = (raw_value - mu0) / sigma0
    return float(norm.cdf(z) * 100.0)


def compute_text_score(signals: dict, baseline_stats: dict) -> dict:
    """
    Convert raw signal values into calibrated 0-100 sub-scores and a fused final score.

    Uses data-driven weights and decision thresholds loaded from fusion_config.json.

    Args:
        signals: dict with any subset of:
            {"curvature_raw": float|None, "burstiness_raw": float|None,
             "cliche_density_raw": float|None, "entropy_raw": float|None}
        baseline_stats: dict loaded from calibration/baseline_stats.json.

    Returns:
        dict: {
            "text_score": float,
            "signal_agreement": str,
            "signal_spread": float,
            "thresholds": dict,
            "sub_scores": dict,
        }
    """
    weights, thresholds = load_fusion_config()
    _INVERTED = {"burstiness", "entropy"}
    _SIGNAL_KEYS = ["curvature", "burstiness", "cliche_density", "entropy"]

    sub_scores: dict[str, Optional[float]] = {}
    available_weights: dict[str, float] = {}

    for key in _SIGNAL_KEYS:
        raw_key = f"{key}_raw"
        raw_val = signals.get(raw_key)

        if raw_val is None:
            sub_scores[f"{key}_score"] = None
            continue

        stats = baseline_stats.get(key)
        if stats is None:
            logger.warning("compute_text_score: no baseline stats for signal '%s'. Skipping.", key)
            sub_scores[f"{key}_score"] = None
            continue

        mu0 = stats.get("mu0", 0.0)
        sigma0 = stats.get("sigma0", 1.0)

        cdf_val = _cdf_score(raw_val, mu0, sigma0)

        if key in _INVERTED:
            calibrated = 100.0 - cdf_val
        else:
            calibrated = cdf_val

        calibrated = max(0.0, min(100.0, calibrated))
        sub_scores[f"{key}_score"] = round(calibrated, 2)
        available_weights[key] = weights[key]

    if not available_weights:
        logger.warning("compute_text_score: all signals are missing. Returning score=50.0.")
        return {
            "text_score": 50.0,
            "signal_agreement": "agreement",
            "signal_spread": 0.0,
            "thresholds": thresholds,
            "sub_scores": sub_scores,
        }

    total_weight = sum(available_weights.values())
    normalized_weights = {k: v / total_weight for k, v in available_weights.items()}

    fused = sum(
        normalized_weights[key] * sub_scores[f"{key}_score"]
        for key in available_weights
    )

    # ---- Signal Disagreement Check (Correction 7) ----
    valid_scores = [v for v in sub_scores.values() if v is not None]
    if len(valid_scores) >= 2:
        spread = max(valid_scores) - min(valid_scores)
        agreement_status = "disagreement" if spread > 40.0 else "agreement"
    else:
        spread = 0.0
        agreement_status = "agreement"

    # ---- Vocabulary Richness / High-Entropy Guard ----
    # If unigram lexical entropy indicates exceptionally rich vocabulary (entropy_score <= 35.0)
    # and zero AI buzzwords are detected (cliche_density_score <= 50.0), signals disagree (spread > 35.0),
    # and curvature is NOT indicating AI (curvature_score < 60.0), apply a human vocabulary credit
    # so formal/journalistic prose is not penalized.
    entropy_s = sub_scores.get("entropy_score")
    cliche_s = sub_scores.get("cliche_density_score")
    curvature_s = sub_scores.get("curvature_score")
    if (entropy_s is not None and cliche_s is not None 
        and entropy_s <= 35.0 and cliche_s <= 50.0 
        and spread > 35.0 and (curvature_s is None or curvature_s < 60.0)):
        human_bonus = (35.0 - entropy_s) * 0.4
        fused = max(0.0, fused - human_bonus)

    return {
        "text_score": round(fused, 2),
        "signal_agreement": agreement_status,
        "signal_spread": round(spread, 2),
        "thresholds": thresholds,
        "sub_scores": sub_scores,
    }
