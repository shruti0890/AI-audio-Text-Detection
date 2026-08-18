"""
text_forensics/fusion.py

Signal Fusion module — Production Five-Feature Architecture.

PATH A — Four-Feature Corrected Baseline (legacy fallback):
    Converts raw signal values (curvature, burstiness, cliche_density, entropy) into
    a single calibrated 0-100 AI-likelihood score using Gaussian CDF mapping.
    Corrected weights: curvature=0.65, burstiness=0.20, cliche=0.05, entropy=0.10.

PATH B — Five-Feature Logistic Regression (PRODUCTION):
    Loads the trained Logistic Regression + StandardScaler from
    calibration/five_feature_model/. Returns an AI probability in [0, 1].
    Features: [curvature, burstiness, lexical_entropy, structural_regularity, cliche_density]
    Calibrated Decision Thresholds:
        Human:        P(AI) <= 0.20
        Likely Human: 0.20 < P(AI) < 0.45
        Likely AI:    0.45 <= P(AI) < 0.70
        AI:           P(AI) >= 0.70

    Falls back to PATH A if the model file is absent.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

from scipy.stats import norm

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "calibration" / "fusion_config.json"
_FIVE_FEATURE_MODEL_DIR = Path(__file__).parent / "calibration" / "five_feature_model"

# Corrected baseline fallback weights (Correction 15)
_DEFAULT_WEIGHTS = {
    "curvature": 0.65,
    "burstiness": 0.20,
    "cliche_density": 0.05,
    "entropy": 0.10,
}

_DEFAULT_THRESHOLDS = {
    "human_max": 69.7443,
    "mixed_range": [69.7443, 77.996],
    "ai_min": 77.996,
}

# Production 4-way calibrated probability thresholds (0.0 to 1.0)
_DEFAULT_LR_THRESHOLDS_4WAY = {
    "human_max": 0.20,
    "likely_human_max": 0.45,
    "likely_ai_min": 0.45,
    "ai_min": 0.70,
}

# Module-level cache for the five-feature model
_FIVE_FEATURE_MODEL = None
_FIVE_FEATURE_SCALER = None
_FIVE_FEATURE_METADATA = None
_MODEL_LOAD_ATTEMPTED = False


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
                "burstiness": float(raw_w.get("burstiness", 0.20)),
                "cliche_density": float(raw_w.get("cliche", raw_w.get("cliche_density", 0.05))),
                "entropy": float(raw_w.get("entropy", 0.10)),
            }
            w_sum = sum(weights.values())
            if w_sum > 0:
                weights = {k: v / w_sum for k, v in weights.items()}

            thresholds = data.get("thresholds", _DEFAULT_THRESHOLDS)
            return weights, thresholds
        except Exception as e:
            logger.warning("Failed to load %s (%s). Using default weights.", _CONFIG_PATH, e)

    return _DEFAULT_WEIGHTS, _DEFAULT_THRESHOLDS


def load_five_feature_model() -> Tuple[Optional[object], Optional[object], Optional[dict]]:
    """
    Load the trained production five-feature Logistic Regression model, scaler, and metadata.

    Models are cached at module level after first load.

    Returns:
        tuple: (model, scaler, metadata)
               Any element may be None if the model file is absent or load fails.
    """
    global _FIVE_FEATURE_MODEL, _FIVE_FEATURE_SCALER, _FIVE_FEATURE_METADATA, _MODEL_LOAD_ATTEMPTED

    if _MODEL_LOAD_ATTEMPTED:
        return _FIVE_FEATURE_MODEL, _FIVE_FEATURE_SCALER, _FIVE_FEATURE_METADATA

    _MODEL_LOAD_ATTEMPTED = True
    model_path = _FIVE_FEATURE_MODEL_DIR / "model.pkl"
    scaler_path = _FIVE_FEATURE_MODEL_DIR / "scaler.pkl"
    meta_path = _FIVE_FEATURE_MODEL_DIR / "model_metadata.json"

    if not model_path.exists():
        logger.info(
            "Five-feature model not found at %s. "
            "Falling back to four-feature corrected baseline.",
            model_path,
        )
        return None, None, None

    try:
        with open(model_path, "rb") as f:
            _FIVE_FEATURE_MODEL = pickle.load(f)
        with open(scaler_path, "rb") as f:
            _FIVE_FEATURE_SCALER = pickle.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            _FIVE_FEATURE_METADATA = json.load(f)
        logger.info("Loaded production five-feature LR model from %s", _FIVE_FEATURE_MODEL_DIR)
    except Exception as e:
        logger.warning("Failed to load five-feature model: %s. Using baseline.", e)
        _FIVE_FEATURE_MODEL = None
        _FIVE_FEATURE_SCALER = None
        _FIVE_FEATURE_METADATA = None

    return _FIVE_FEATURE_MODEL, _FIVE_FEATURE_SCALER, _FIVE_FEATURE_METADATA


# Alias for backward compatibility
load_six_feature_model = load_five_feature_model


def _cdf_score(raw_value: float, mu0: float, sigma0: float) -> float:
    """Map a raw signal value to a calibrated 0-100 sub-score via Gaussian CDF."""
    if sigma0 <= 0:
        logger.warning("sigma0 <= 0 for signal; defaulting sub-score to 50.0")
        return 50.0
    z = (raw_value - mu0) / sigma0
    return float(norm.cdf(z) * 100.0)


def compute_text_score(signals: dict, baseline_stats: dict) -> dict:
    """
    PATH A — Four-Feature Corrected Baseline.

    Convert raw signal values into calibrated 0-100 sub-scores and a fused final score.
    Uses data-driven weights (0.65/0.20/0.05/0.10).

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

    # ---- Signal Disagreement Check ----
    valid_scores = [v for v in sub_scores.values() if v is not None]
    if len(valid_scores) >= 2:
        spread = max(valid_scores) - min(valid_scores)
        agreement_status = "disagreement" if spread > 40.0 else "agreement"
    else:
        spread = 0.0
        agreement_status = "agreement"

    # ---- Vocabulary Richness / High-Entropy Guard ----
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


def compute_five_feature_score(
    curvature: Optional[float],
    burstiness: Optional[float],
    lexical_entropy: Optional[float],
    structural_regularity: Optional[float],
    cliche_density: Optional[float],
    baseline_stats: dict,
) -> dict:
    """
    PATH B — Five-Feature Logistic Regression (PRODUCTION).

    Applies the trained StandardScaler + LogisticRegression model from
    five_feature_model/ to produce an AI probability.

    Exact canonical feature order:
        ["curvature", "burstiness", "lexical_entropy", "structural_regularity", "cliche_density"]

    Args:
        curvature:              Raw curvature discrepancy d(x) or None
        burstiness:             Raw sigma/mu ratio or None
        lexical_entropy:        Raw Shannon entropy (bits) or None
        structural_regularity:  Raw structural composite [0,100] or None
        cliche_density:         Cliche density % [0,100]
        baseline_stats:         dict from baseline_stats.json (for CDF fallback)

    Returns:
        dict: {
            "ai_probability":  float,    # LR output P(AI|X) in [0, 1]
            "ai_score":        float,    # ai_probability * 100 in [0, 100]
            "verdict":         str,      # "Human" | "Likely Human" | "Likely AI" | "AI"
            "confidence":      str,      # "Low" | "Moderate" | "High"
            "model_used":      str,      # "five_feature_logistic_regression" or "corrected_four_feature_baseline"
            "thresholds_used": dict,     # probability thresholds applied
        }
    """
    model, scaler, metadata = load_five_feature_model()

    if model is None or scaler is None or metadata is None:
        logger.info("compute_five_feature_score: model not available — using baseline score as fallback.")
        raw_signals = {
            "curvature_raw": curvature,
            "burstiness_raw": burstiness,
            "cliche_density_raw": cliche_density,
            "entropy_raw": lexical_entropy,
        }
        baseline_result = compute_text_score(raw_signals, baseline_stats)
        text_score = baseline_result["text_score"]
        ai_prob = text_score / 100.0
        thresholds_4way = _DEFAULT_LR_THRESHOLDS_4WAY
        model_used = "corrected_four_feature_baseline"
    else:
        # Build 5-element feature vector in exact canonical order
        raw_vector = [
            curvature,
            burstiness,
            lexical_entropy,
            structural_regularity,
            cliche_density,
        ]

        feature_order = metadata.get("feature_order", [
            "curvature", "burstiness", "lexical_entropy",
            "structural_regularity", "cliche_density"
        ])
        col_means = metadata.get("feature_means", {})

        imputed = []
        for fname, val in zip(feature_order, raw_vector):
            if val is None:
                fallback_val = col_means.get(fname, 0.0)
                logger.info(
                    "compute_five_feature_score: feature '%s' is None — imputing with training mean %.4f",
                    fname,
                    fallback_val,
                )
                imputed.append(fallback_val)
            else:
                imputed.append(val)

        import numpy as np
        X = np.array(imputed, dtype=float).reshape(1, -1)
        X_scaled = scaler.transform(X)

        proba = model.predict_proba(X_scaled)[0]
        ai_prob = float(proba[1])

        thresholds_4way = metadata.get("thresholds_4way", _DEFAULT_LR_THRESHOLDS_4WAY)
        model_used = "five_feature_logistic_regression"

    # ---- Calibrated 4-Way Decision Classification ----
    # Boundaries:
    #   P(AI) <= 0.20        -> Human
    #   0.20 < P(AI) < 0.45  -> Likely Human
    #   0.45 <= P(AI) < 0.70 -> Likely AI
    #   P(AI) >= 0.70        -> AI
    t_ai = thresholds_4way.get("ai_min", 0.70)
    t_likely_ai = thresholds_4way.get("likely_ai_min", 0.45)
    t_human = thresholds_4way.get("human_max", 0.20)

    if ai_prob >= t_ai:
        verdict = "AI"
        confidence = "High" if ai_prob >= 0.85 else "Moderate"
    elif ai_prob >= t_likely_ai:
        verdict = "Likely AI"
        confidence = "Moderate"
    elif ai_prob <= t_human:
        verdict = "Human"
        confidence = "High" if ai_prob <= 0.10 else "Moderate"
    else:
        verdict = "Likely Human"
        confidence = "Low"

    return {
        "ai_probability": round(ai_prob, 4),
        "ai_score": round(ai_prob * 100.0, 2),
        "verdict": verdict,
        "confidence": confidence,
        "model_used": model_used,
        "thresholds_used": thresholds_4way,
    }


def compute_six_feature_score(
    curvature: Optional[float],
    burstiness: Optional[float],
    lexical_entropy: Optional[float],
    ngram_repetition: Optional[float] = None,
    structural_regularity: Optional[float] = None,
    cliche_density: Optional[float] = None,
    baseline_stats: dict = None,
) -> dict:
    """Backward-compatible wrapper delegating to production compute_five_feature_score."""
    return compute_five_feature_score(
        curvature=curvature,
        burstiness=burstiness,
        lexical_entropy=lexical_entropy,
        structural_regularity=structural_regularity,
        cliche_density=cliche_density,
        baseline_stats=baseline_stats or {},
    )
