"""
text_forensics/calibration/generator_attribution_experiment/attribution_pipeline.py

Two-Stage Unified Forensics & Attribution Pipeline:
  Stage 1: Existing 5-Feature Forensic Detector (Human vs. AI)
  Stage 2: Experimental Multi-Class Generator Attribution (ChatGPT vs Gemini vs Claude vs Other AI)

Non-destructive: Integrates cleanly on top of production analyze_text().
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))

from text_forensics.pipeline import analyze_text
from text_forensics.calibration.generator_attribution_experiment.features.attribution_features import (
    ATTRIBUTION_FEATURE_ORDER,
    extract_attribution_features,
    extract_feature_vector,
)

_MODEL_PATH = _HERE / "models" / "attribution_model.pkl"
_SCALER_PATH = _HERE / "scalers" / "attribution_scaler.pkl"
_CONFIG_PATH = _HERE / "configs" / "attribution_metadata.json"

_MODEL = None
_SCALER = None
_METADATA = None
_ATTRIBUTION_LOADED = False

CLASSES = ["ChatGPT", "Gemini", "Claude", "Other_AI"]


def load_attribution_engine() -> Tuple[Optional[Any], Optional[Any], Optional[Dict]]:
    """Loads and caches the attribution model and scaler."""
    global _MODEL, _SCALER, _METADATA, _ATTRIBUTION_LOADED

    if _ATTRIBUTION_LOADED:
        return _MODEL, _SCALER, _METADATA

    _ATTRIBUTION_LOADED = True
    if not _MODEL_PATH.exists() or not _SCALER_PATH.exists():
        logger.warning("Attribution model or scaler not found in %s", _HERE)
        return None, None, None

    try:
        with open(_MODEL_PATH, "rb") as f:
            _MODEL = pickle.load(f)
        with open(_SCALER_PATH, "rb") as f:
            _SCALER = pickle.load(f)
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _METADATA = json.load(f)
        logger.info("Loaded generator attribution engine successfully.")
    except Exception as e:
        logger.warning("Failed to load attribution engine: %s", e)
        _MODEL, _SCALER, _METADATA = None, None, None

    return _MODEL, _SCALER, _METADATA


def analyze_text_with_attribution(text: str, run_robustness: bool = False) -> Dict[str, Any]:
    """
    Executes the full two-stage forensic and generator attribution analysis.

    Stage 1: Evaluates binary Human/AI probability with calibrated 4-way verdict.
    Stage 2: If Likely AI / AI, estimates generator attribution distribution.
             If Human / Likely Human, explicitly bypasses generator attribution.

    Args:
        text: Non-empty string.
        run_robustness: Whether to execute T5 adversarial robustness check.

    Returns:
        Dict containing all production forensic keys + Stage 2 attribution keys.
    """
    # ── STAGE 1: Production Human vs AI Forensic Detection ─────────────────
    result = analyze_text(text, run_robustness=run_robustness)
    verdict = result.get("verdict", "Human")
    ai_prob = result.get("ai_probability", 0.0)

    # ── STAGE 2: Conditional AI-Generator Attribution ──────────────────────
    if verdict in ["Human", "Likely Human"]:
        result["generator_attribution"] = None
        result["predicted_generator"] = "Not applicable"
        result["generator_confidence"] = 0.0
        result["generator_verdict"] = "Not applicable / Insufficient AI evidence"
        result["attribution_disclaimer"] = "Generator attribution is bypassed when text is classified as Human."
        return result

    # For Likely AI / AI: Run generator attribution model
    model, scaler, metadata = load_attribution_engine()

    if model is None or scaler is None:
        result["generator_attribution"] = {
            "chatgpt": 0.25,
            "gemini": 0.25,
            "claude": 0.25,
            "other_ai": 0.25,
        }
        result["predicted_generator"] = "Unknown / Other AI"
        result["generator_confidence"] = 0.25
        result["generator_verdict"] = "Insufficient evidence (Model Unavailable)"
        result["attribution_disclaimer"] = (
            "Generator attribution is probabilistic and should not be interpreted as proof of authorship."
        )
        return result

    # Extract attribution features, passing pre-extracted base signals for efficiency
    base_signals = {
        "curvature": result.get("features", {}).get("curvature"),
        "burstiness": result.get("features", {}).get("burstiness"),
        "lexical_entropy": result.get("features", {}).get("lexical_entropy"),
        "structural_regularity": result.get("features", {}).get("structural_regularity"),
        "cliche_density": result.get("features", {}).get("cliche_density"),
    }
    feat_dict = extract_attribution_features(text, base_signals=base_signals)
    feat_vec = extract_feature_vector(feat_dict)

    X = np.array(feat_vec, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(X_scaled)[0]
    
    chatgpt_p = float(probabilities[0])
    gemini_p = float(probabilities[1])
    claude_p = float(probabilities[2])
    other_p = float(probabilities[3])

    attr_dist = {
        "chatgpt": round(chatgpt_p, 4),
        "gemini": round(gemini_p, 4),
        "claude": round(claude_p, 4),
        "other_ai": round(other_p, 4),
    }

    max_idx = int(np.argmax(probabilities))
    max_p = float(probabilities[max_idx])
    pred_class = CLASSES[max_idx]

    uncertainty_cfg = metadata.get("uncertainty_thresholds", {}) if metadata else {}
    min_confidence = uncertainty_cfg.get("min_confidence_for_attribution", 0.45)

    if max_p < min_confidence:
        predicted_gen = "Unknown / Other AI"
        gen_verdict = "Insufficient evidence"
        gen_conf = round(max_p, 4)
    else:
        predicted_gen = pred_class
        gen_verdict = f"Possible {pred_class}"
        gen_conf = round(max_p, 4)

    result["generator_attribution"] = attr_dist
    result["predicted_generator"] = predicted_gen
    result["generator_confidence"] = gen_conf
    result["generator_verdict"] = gen_verdict
    result["attribution_disclaimer"] = (
        "Generator attribution is probabilistic and should not be interpreted as proof of authorship or identification of the exact model used."
    )

    return result
