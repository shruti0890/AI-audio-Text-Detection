"""
cross_modal_reasoner.py
=======================
Deterministic Cross-Modality Reasoner comparing Audio Forensic Classification
and Transcribed Text Forensic Classification.

Primary States:
  - HUMAN_HUMAN:                    Audio = HUMAN, Text = HUMAN (Modality Consistent)
  - AI_AI:                          Audio = AI,    Text = AI    (Modality Consistent)
  - HUMAN_VOICE_AI_TEXT_CONFLICT:   Audio = HUMAN, Text = AI    (Modality Conflict)
  - AI_VOICE_HUMAN_TEXT_CONFLICT:   Audio = AI,    Text = HUMAN (Modality Conflict)

Classification Rule:
  Audio is classified as 'AI' if audio_score >= decision_threshold (default: 56.0%), else 'HUMAN'.
  Text is classified as 'AI' if text_score (P(AI)*100) >= text_decision_threshold (default: 45.0%), else 'HUMAN'.
"""

from typing import Any, Dict, Optional


def evaluate_cross_modality(
    audio_result: Dict[str, Any],
    text_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministically compares audio forensic analysis with text forensic analysis.

    Args:
        audio_result: Dict containing at least 'audio_score' and optionally 'decision_threshold_pct'.
        text_result: Dict containing at least 'ai_score' (or 'text_score') and optionally 'verdict'.

    Returns:
        Dict with keys:
            - classification: 'HUMAN_HUMAN' | 'AI_AI' | 'HUMAN_VOICE_AI_TEXT_CONFLICT' | 'AI_VOICE_HUMAN_TEXT_CONFLICT'
            - consistent: bool (True if HUMAN_HUMAN or AI_AI)
            - conflict: bool (True if modality mismatch)
            - title: str human-readable UI header
            - description: str human-readable explanation
            - audio: dict with score, threshold, binary classification, detailed verdict
            - text: dict with score, threshold, binary classification, detailed verdict
    """
    # ── 1. Audio Evaluation ───────────────────────────────────────────────────
    audio_score = float(audio_result.get("audio_score", 0.0))
    # fusion_integration stores the threshold as "audio_decision_threshold_pct"
    audio_threshold = float(
        audio_result.get("audio_decision_threshold_pct")
        or audio_result.get("decision_threshold_pct")
        or 56.0
    )
    audio_verdict = audio_result.get("audio_verdict") or ("AI Voice" if audio_score >= audio_threshold else "Human Voice")

    # Binary Audio Classification: "AI" vs "HUMAN"
    is_audio_ai = audio_score >= audio_threshold
    audio_classification = "AI" if is_audio_ai else "HUMAN"

    # ── 2. Text Evaluation ────────────────────────────────────────────────────
    # Production text score is ai_score (ai_probability * 100 in [0, 100])
    text_score = float(text_result.get("ai_score", text_result.get("text_score", 0.0)))
    text_verdict = text_result.get("verdict", text_result.get("text_verdict", "Human"))

    # Five-Feature LR 4-way tiers: "AI" (>=70%) and "Likely AI" (>=45%) → AI bucket
    # "Human" (<=20%) and "Likely Human" (<45%) → HUMAN bucket.
    # Use verdict string as source of truth (already computed by the model against calibrated thresholds).
    verdict_lower = text_verdict.lower()
    is_text_ai = "likely ai" in verdict_lower or (verdict_lower == "ai")
    text_classification = "AI" if is_text_ai else "HUMAN"
    # Threshold shown to user is the boundary between HUMAN and AI buckets (45%)
    text_threshold = 45.0

    # ── 3. Deterministic Cross-Modal State ─────────────────────────────────────
    if audio_classification == "HUMAN" and text_classification == "HUMAN":
        classification = "HUMAN_HUMAN"
        consistent = True
        conflict = False
        title = "✓ MODALITIES CONSISTENT"
        description = "Human Voice + Human Text — Spoken voice characteristics and transcribed linguistic patterns both indicate authentic human creation."

    elif audio_classification == "AI" and text_classification == "AI":
        classification = "AI_AI"
        consistent = True
        conflict = False
        title = "✓ MODALITIES CONSISTENT"
        description = "AI Voice + AI Text — Both speech acoustics (synthetic voice synthesis) and linguistic style (AI language model) indicate artificial generation."

    elif audio_classification == "HUMAN" and text_classification == "AI":
        classification = "HUMAN_VOICE_AI_TEXT_CONFLICT"
        consistent = False
        conflict = True
        title = "⚠️ MODALITY CONFLICT"
        description = "Human Voice + AI Text — Natural human voice detected, but the spoken text was likely drafted by an AI language model (e.g. human reading an LLM script)."

    else:  # audio_classification == "AI" and text_classification == "HUMAN"
        classification = "AI_VOICE_HUMAN_TEXT_CONFLICT"
        consistent = False
        conflict = True
        title = "⚠️ MODALITY CONFLICT"
        description = "AI Voice + Human Text — Synthetic voice/TTS detected, but the underlying text exhibits human writing patterns (e.g. voice cloning of human-written text)."

    return {
        "classification": classification,
        "consistent": consistent,
        "conflict": conflict,
        "title": title,
        "description": description,
        "audio": {
            "score": round(audio_score, 2),
            "threshold": round(audio_threshold, 1),
            "classification": audio_classification,
            "verdict": audio_verdict,
        },
        "text": {
            "score": round(text_score, 2),
            "threshold": round(text_threshold, 1),
            "classification": text_classification,
            "verdict": text_verdict,
        },
    }
