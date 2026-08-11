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


# ── Audio badge thresholds ───────────────────────────────────────────────────
# [UNCALIBRATED PLACEHOLDER]
# Source: architecture doc illustrative values.
# These numbers have NOT been validated against ground-truth audio data.
# Treat them the same way text's thresholds were treated BEFORE Correction 9:
#   as a starting assumption, clearly labelled, subject to a future calibration
#   pass analogous to the text weight grid-search.
_AUDIO_AI_MIN: float = 75.0        # score >= this → "Likely AI / Deepfake"
_AUDIO_HUMAN_MAX: float = 25.0     # score <  this → "Likely Human / Real"
# 25 ≤ score < 75 → "Inconclusive"

_AUDIO_VERDICT_CAVEAT = (
    "[UNCALIBRATED PLACEHOLDER] Audio badge thresholds (≥75 AI, <25 Human, 25-75 Inconclusive) "
    "are illustrative values from the architecture document and have NOT been validated "
    "against ground-truth audio data. They will be replaced after a Correction-9-style "
    "calibration run on matched real/fake audio clips."
)

# ── Text badge thresholds ─────────────────────────────────────────────────────
# Calibrated via Correction 9 grid-search (ROC-AUC 0.9994 on 120 HC3 samples).
_TEXT_AI_MIN: float = 75.0
_TEXT_MIXED_MIN: float = 50.0

# ── Unified score note ────────────────────────────────────────────────────────
_UNIFIED_SCORE_NOTE = (
    "Unified score is an equal-weight average of text_score and audio_score "
    "(placeholder — recalibrate weights after audio thresholds are validated)."
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _text_verdict(score: float) -> str:
    """Assign a text verdict badge using calibrated Correction 9 thresholds."""
    if score >= _TEXT_AI_MIN:
        return "Likely AI-Generated"
    if score >= _TEXT_MIXED_MIN:
        return "Uncertain / Mixed Signals"
    return "Likely Human-Written"


def _audio_verdict(score: float) -> str:
    """Assign an audio verdict badge using UNCALIBRATED PLACEHOLDER thresholds."""
    if score >= _AUDIO_AI_MIN:
        return "Likely AI / Deepfake [UNCALIBRATED]"
    if score >= _AUDIO_HUMAN_MAX:
        return "Inconclusive [UNCALIBRATED]"
    return "Likely Human / Real Voice [UNCALIBRATED]"


def _unified_verdict(score: float) -> str:
    """Assign a unified verdict. Uses text thresholds as a reasonable proxy."""
    if score >= _TEXT_AI_MIN:
        return "Likely AI-Generated (Combined)"
    if score >= _TEXT_MIXED_MIN:
        return "Uncertain / Mixed (Combined)"
    return "Likely Human (Combined)"


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
            result["text_score"]            = text_out["text_score"]
            result["text_verdict"]          = _text_verdict(text_out["text_score"])
            result["text_signals"]          = text_out.get("signals", {})
            result["text_signal_agreement"] = text_out.get("signal_agreement", "agreement")
            result["text_stability_flag"]   = text_out.get("stability_flag", "skipped")
            logger.info("[fusion] Text score: %.2f → %s", text_out["text_score"], result["text_verdict"])
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
            logger.info(
                "[fusion] Audio score: %.2f → %s", audio_out["audio_score"], result["audio_verdict"]
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
