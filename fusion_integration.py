"""
fusion_integration.py
=====================
Integration Phase — Text + Audio Forensics Unified Fusion Layer.

This is the single entry point for the combined system. It:
  1. Accepts text (str) and/or an audio file path (str | None).
  2. Runs the text forensics pipeline (`text_forensics.pipeline.analyze_text`).
  3. Runs the audio forensics pipeline (`audio_forensics.pipeline.analyze_audio`).
  4. Assigns per-modality verdict badges.
  5. Computes a preliminary Unified AI-Likelihood Score (equal-weight average,
     clearly marked as an uncalibrated placeholder pending audio threshold tuning).
  6. Exposes raw cross-modal delta for future Section 3C calibration (NOT a badge).

Architecture Sections implemented here:
  - Section 1  : Input routing (text / audio / combined)
  - Section 2  : Per-modality pipeline execution
  - Section 3A : Text verdict badge (calibrated — Correction 9 thresholds)
  - Section 3B : Audio verdict badge (UNCALIBRATED PLACEHOLDER thresholds)
  - Section 3C : DEFERRED — cross_modal_deferred=True flag returned, raw delta
                 exposed for future calibration but NO Modality Conflict badge issued.
  - Section 4  : Unified score (placeholder equal-weight average)
  - Section 5  : Unified verdict string

LOCKED OUTPUT SCHEMA — do not rename keys:
{
    # ── Text Modality ────────────────────────────────────────────────────────
    "text_score":          float | None,   # 0-100, None if no text supplied
    "text_verdict":        str   | None,   # Section 3A badge
    "text_signals":        dict  | None,   # full signals sub-dict from text pipeline
    "text_signal_agreement": str | None,   # "agreement" | "disagreement" | None
    "text_stability_flag": str  | None,    # "stable" | "unstable" | "skipped" | None

    # ── Audio Modality ───────────────────────────────────────────────────────
    "audio_score":         float | None,   # 0-100, None if no audio supplied
    "audio_verdict":       str   | None,   # Section 3B badge (UNCALIBRATED PLACEHOLDER)
    "audio_verdict_note":  str,            # explicit caveat string — always present
    "logit_fake":          float | None,
    "logit_real":          float | None,
    "transcript":          str,            # "" if no audio

    # ── Cross-Modal (Section 3C deferred) ────────────────────────────────────
    "cross_modal_deferred":    bool,       # always True — section not implemented
    "transcript_text_delta":   float | None,  # raw |text_score - transcript_score|
                                              # computed for future calibration only

    # ── Unified (Section 4 + 5) ──────────────────────────────────────────────
    "unified_score":       float | None,   # placeholder equal-weight average
    "unified_verdict":     str   | None,   # unified badge string
    "unified_score_note":  str,            # caveat about equal-weight placeholder
}
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Path setup so this file is runnable from the project root ─────────────────
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── Audio badge thresholds — loaded from model_config.json ───────────────────
# These are loaded once at import time so all callers share the same config.
# To recalibrate: update calibration_parameters.threshold_tiers in
# audio_forensics/calibration/model_config.json and restart.
# Calibrated 4-tier system:
#   0–35  Authentic Human Voice
#   36–55 Likely Human Voice
#   56–74 Likely AI Voice
#   75–100 Authentic AI Voice
def _load_audio_badge_thresholds() -> dict:
    """Read calibrated tier boundaries from model_config.json.
    Returns a dict with keys: authentic_human_max, likely_human_max,
    likely_ai_max, authentic_ai_max, and decision_threshold_pct.
    Falls back to calibrated defaults if config is absent.
    """
    import json
    config_path = Path(__file__).resolve().parent / "audio_forensics" / "calibration" / "model_config.json"
    defaults = {
        "authentic_human_max": 35.0,
        "likely_human_max": 55.0,
        "likely_ai_max": 74.0,
        "decision_threshold_pct": 56.0,
    }
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        params = cfg.get("calibration_parameters", {})
        tiers = params.get("threshold_tiers", {})
        return {
            "authentic_human_max": float(tiers.get("authentic_human", [0, 35])[1]),
            "likely_human_max":    float(tiers.get("likely_human",    [36, 55])[1]),
            "likely_ai_max":       float(tiers.get("likely_ai",       [56, 74])[1]),
            "decision_threshold_pct": float(params.get("decision_threshold_pct", 56.0)),
        }
    except Exception:
        return defaults


_AUDIO_TIERS = _load_audio_badge_thresholds()
# Tier boundaries:
#   score <= authentic_human_max (35)  → Authentic Human Voice
#   score <= likely_human_max (55)     → Likely Human Voice
#   score <= likely_ai_max (74)        → Likely AI Voice
#   score >  likely_ai_max (74)        → Authentic AI Voice

_AUDIO_VERDICT_CAVEAT = (
    "Audio forensic thresholds calibrated: 0–35 Authentic Human Voice, "
    "36–55 Likely Human Voice, 56–74 Likely AI Voice, 75–100 Authentic AI Voice."
)

# ── Text badge thresholds ─────────────────────────────────────────────────────
def _load_text_badge_thresholds() -> tuple[float, float, float]:
    """Read 4-way decision thresholds from text_forensics/calibration/fusion_config.json.
    Returns (human_max, likely_human_max, ai_min) boundaries.
      score < human_max             -> "Human"
      human_max <= score < mid_max  -> "Likely Human"
      mid_max <= score < ai_min     -> "Likely AI"
      score >= ai_min               -> "AI"
    """
    import json
    config_path = Path(__file__).resolve().parent / "text_forensics" / "calibration" / "fusion_config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        th = cfg.get("thresholds_4way", {})
        t_human = float(th.get("human_max", 50.0))
        t_mid = float(th.get("likely_human_max", 70.0))
        t_ai = float(th.get("ai_min", 85.0))
        return t_human, t_mid, t_ai
    except Exception:
        return 50.0, 70.0, 85.0


_TEXT_HUMAN_MAX, _TEXT_MID_MAX, _TEXT_AI_MIN = _load_text_badge_thresholds()


# ── Unified score note ────────────────────────────────────────────────────────
_UNIFIED_SCORE_NOTE = (
    "Unified score is an equal-weight average of text_score and audio_score "
    "(placeholder — recalibrate weights after audio thresholds are validated)."
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _text_verdict(score: float) -> str:
    """Assign a 4-way text verdict badge: Human, Likely Human, Likely AI, AI."""
    human_max, mid_max, ai_min = _load_text_badge_thresholds()
    if score >= ai_min:
        return "AI"
    if score >= mid_max:
        return "Likely AI"
    if score >= human_max:
        return "Likely Human"
    return "Human"


def _audio_verdict(score: float) -> str:
    """Assign an audio verdict badge using calibrated threshold tiers:
    0-35: Authentic Human Voice
    36-55: Likely Human Voice
    56-74: Likely AI Voice
    75-100: Authentic AI Voice
    """
    if score >= 75.0:
        return "Authentic AI Voice"
    elif score >= 56.0:
        return "Likely AI Voice"
    elif score >= 36.0:
        return "Likely Human Voice"
    else:
        return "Authentic Human Voice"


def _unified_verdict(score: float) -> str:
    """Assign a 4-way unified verdict."""
    human_max, mid_max, ai_min = _load_text_badge_thresholds()
    if score >= ai_min:
        return "AI (Combined)"
    if score >= mid_max:
        return "Likely AI (Combined)"
    if score >= human_max:
        return "Likely Human (Combined)"
    return "Human (Combined)"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    text: Optional[str] = None,
    audio_path: Optional[str] = None,
    run_robustness: bool = False,
) -> dict:
    """
    Run the combined Text + Audio forensics pipeline.

    At least one of `text` or `audio_path` must be provided.

    Args:
        text:            Plain text string to analyze (or None to skip text modality).
        audio_path:      Path to an audio file (.wav/.mp3/.flac/etc.) or None to skip.
        run_robustness:  Whether to run T5 paraphrase robustness check on text
                         (adds ~30s; default False to keep combined mode snappy).

    Returns:
        dict matching the locked output schema defined in this module's docstring.

    Raises:
        ValueError: If neither text nor audio_path is provided.
        FileNotFoundError: If audio_path is given but the file does not exist.
    """
    if not text and not audio_path:
        raise ValueError(
            "run_full_pipeline() requires at least one of: text (str) or audio_path (str)."
        )

    result: dict = {
        # text defaults
        "text_score":            None,
        "text_verdict":          None,
        "text_signals":          None,
        "text_signal_agreement": None,
        "text_stability_flag":   None,
        # audio defaults
        "audio_score":           None,
        "audio_verdict":         None,
        "audio_verdict_note":    _AUDIO_VERDICT_CAVEAT,
        "logit_fake":            None,
        "logit_real":            None,
        "transcript":            "",
        "audio_vad_time":        None,
        "audio_asr_time":        None,
        "audio_deepfake_time":   None,
        "audio_total_time":      None,
        # cross-modal
        "cross_modal_deferred":  True,
        "transcript_text_delta": None,
        # unified
        "unified_score":         None,
        "unified_verdict":       None,
        "unified_score_note":    _UNIFIED_SCORE_NOTE,
    }

    # ── Section 2 / 3A: Text pipeline ─────────────────────────────────────────
    if text and text.strip():
        logger.info("[fusion] Running text forensics pipeline...")
        try:
            from text_forensics.pipeline import analyze_text
            text_out = analyze_text(text.strip(), run_robustness=run_robustness)
            result["text_score"]            = text_out["ai_score"]  # Production P(AI) * 100 in [0, 100]
            result["text_ai_probability"]   = text_out["ai_probability"]  # P(AI) in [0, 1]
            result["text_verdict"]          = text_out["verdict"]  # Calibrated 4-way verdict
            result["text_confidence"]       = text_out.get("confidence", "Moderate")
            result["text_model_used"]       = text_out.get("model_used", "five_feature_logistic_regression")
            result["text_features"]         = text_out.get("features", {})
            result["text_sentence_evidence"] = text_out.get("sentence_evidence", {})
            result["text_legacy_score"]     = text_out["text_score"]  # Legacy PATH A score for debug
            result["text_signals"]          = text_out.get("signals", {})
            result["text_signal_agreement"] = text_out.get("signal_agreement", "agreement")
            result["text_stability_flag"]   = text_out.get("stability_flag", "skipped")
            logger.info(
                "[fusion] Text AI Prob: %.4f (%.2f%%) → %s [Model: %s]",
                text_out["ai_probability"], text_out["ai_score"], result["text_verdict"], result["text_model_used"]
            )
        except Exception as exc:
            logger.error("[fusion] Text pipeline failed: %s", exc, exc_info=True)
            raise

    # ── Section 2 / 3B: Audio pipeline ────────────────────────────────────────
    if audio_path:
        logger.info("[fusion] Running audio forensics pipeline...")
        try:
            from audio_forensics.pipeline import analyze_audio
            audio_out = analyze_audio(audio_path)
            result["audio_score"]   = audio_out["audio_score"]
            result["audio_verdict"] = _audio_verdict(audio_out["audio_score"])
            result["logit_fake"]    = audio_out["logit_fake"]
            result["logit_real"]    = audio_out["logit_real"]
            result["transcript"]    = audio_out.get("transcript", "")
            # Pass calibration metadata through so UI can display them
            result["audio_decision_threshold_pct"] = audio_out.get("decision_threshold_pct", _AUDIO_TIERS["decision_threshold_pct"])
            result["audio_temperature"]            = audio_out.get("temperature", 1.15)
            result["audio_n_windows"]              = audio_out.get("n_windows", 1)
            result["audio_window_scores"]          = audio_out.get("window_scores", [])
            result["audio_confidence_tiers"]       = audio_out.get("confidence_tiers", {})
            result["audio_vad_time"]               = audio_out.get("vad_time")
            result["audio_asr_time"]               = audio_out.get("asr_time")
            result["audio_deepfake_time"]          = audio_out.get("deepfake_time")
            result["audio_total_time"]             = audio_out.get("total_time")
            logger.info(
                "[fusion] Audio score: %.2f → %s (threshold=%.1f%%, T=%.2f, windows=%d)",
                audio_out["audio_score"], result["audio_verdict"],
                result["audio_decision_threshold_pct"], result["audio_temperature"],
                result["audio_n_windows"],
            )
        except Exception as exc:
            logger.error("[fusion] Audio pipeline failed: %s", exc, exc_info=True)
            raise

    # ── Section 3C: Cross-modal delta (DEFERRED — raw only, no badge) ─────────
    # Compute transcript score against original text for future threshold calibration.
    # NO Modality Conflict badge is issued here. cross_modal_deferred=True always.
    transcript = result["transcript"]
    if transcript.strip() and result["text_score"] is not None:
        try:
            from text_forensics.pipeline import analyze_text
            transcript_result = analyze_text(transcript.strip(), run_robustness=False)
            transcript_score = transcript_result["text_score"]
            delta = abs(result["text_score"] - transcript_score)
            result["transcript_text_delta"] = round(delta, 2)
            logger.info(
                "[fusion] Cross-modal delta (deferred, raw only): %.2f "
                "(text_score=%.2f, transcript_score=%.2f)",
                delta, result["text_score"], transcript_score,
            )
        except Exception as exc:
            logger.warning("[fusion] Cross-modal delta computation failed: %s", exc)
            result["transcript_text_delta"] = None
    elif transcript.strip() and result["text_score"] is None:
        # Audio-only mode: score the transcript as the text signal
        try:
            from text_forensics.pipeline import analyze_text
            transcript_result = analyze_text(transcript.strip(), run_robustness=False)
            # Store transcript text score in result for audio-only mode
            result["transcript_text_delta"] = None  # no original text to delta against
            logger.info(
                "[fusion] Audio-only mode: transcript text score = %.2f",
                transcript_result["text_score"],
            )
        except Exception:
            pass

    # ── Section 4: Unified score (placeholder equal-weight average) ────────────
    available_scores = [s for s in [result["text_score"], result["audio_score"]] if s is not None]
    if available_scores:
        unified = sum(available_scores) / len(available_scores)
        result["unified_score"]   = round(unified, 2)
        result["unified_verdict"] = _unified_verdict(unified)
        logger.info(
            "[fusion] Unified score: %.2f → %s (n=%d modalities)",
            unified, result["unified_verdict"], len(available_scores),
        )

    return result
