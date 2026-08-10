"""
Audio Forensics Pipeline Entrypoint
====================================
Phase 2.5 — Signal Log & Output Schema

Main pipeline exposing `analyze_audio(audio_path)` to the top-level fusion layer.
Chains Voice Activity Detection (VAD) → ASR (Whisper-Tiny) → Deepfake Scoring (wav2vec2)
and assembles a standardised output schema dict for consumption by fusion.py and the
Cross-Modal Consistency Check.

Output schema (keys are contractually fixed — do not rename):
    {
        "audio_score":         float,  # deepfake confidence 0.0 (real) … 100.0 (fake)
        "logit_fake":          float,  # raw pre-softmax logit for the 'fake' class
        "logit_real":          float,  # raw pre-softmax logit for the 'real' class
        "transcript":          str,    # spoken text from Whisper-Tiny (used by text_forensics & cross-modal check)
        "wer_confidence_note": str,    # fixed human-readable caveat for fusion consumers
    }
"""

import os
import time
from typing import Dict, Any

from .vad import strip_silence
from .asr import transcribe
from .deepfake_model import score_audio


def analyze_audio(audio_path: str) -> Dict[str, Any]:
    """Runs the full audio forensics pipeline on an input file.

    Chains:
        strip_silence(audio_path) → transcribe(waveform) → score_audio(waveform)

    Args:
        audio_path (str): Path to input audio clip (.wav, .mp3, etc.).
                          Must be an existing file; FileNotFoundError is raised otherwise.

    Returns:
        Dict[str, Any]: Audio forensic evaluation results formatted for fusion consumption.
            Key names are contractually fixed and must not be changed.

            Example output:
                {
                    "audio_score":         81.2,
                    "logit_fake":          2.14,
                    "logit_real":          -0.87,
                    "transcript":          "text transcribed from audio...",
                    "wer_confidence_note": "internal sanity-check only, not per-clip WER"
                }

    Raises:
        FileNotFoundError: If `audio_path` does not exist on disk.
        ValueError: If the audio input type is not supported by downstream modules.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"[pipeline] Audio file not found: {audio_path}")

    t_start = time.perf_counter()
    print(f"[pipeline] Starting analysis -> {audio_path}")

    # ── 1. Voice Activity Detection (Silence Stripping) ───────────────────────
    t0 = time.perf_counter()
    processed_audio = strip_silence(audio_path)
    print(f"[pipeline] VAD complete  | samples={len(processed_audio):,} "
          f"| duration={len(processed_audio)/16000:.2f}s "
          f"| elapsed={time.perf_counter()-t0:.2f}s")

    # ── 2. Speech-to-Text Transcription (Whisper-Tiny) ────────────────────────
    t0 = time.perf_counter()
    transcript = transcribe(processed_audio)
    print(f"[pipeline] ASR complete  | chars={len(transcript)} "
          f"| elapsed={time.perf_counter()-t0:.2f}s")
    print(f"[pipeline] Transcript    | \"{transcript[:120]}{'...' if len(transcript) > 120 else ''}\"")

    # ── 3. Deepfake Classification (wav2vec2-deepfake-voice-detector) ─────────
    t0 = time.perf_counter()
    scores = score_audio(processed_audio)
    print(f"[pipeline] Deepfake done | s_audio={scores['s_audio']:.4f} "
          f"logit_fake={scores['logit_fake']:.6f} logit_real={scores['logit_real']:.6f} "
          f"| elapsed={time.perf_counter()-t0:.2f}s")

    total = time.perf_counter() - t_start
    print(f"[pipeline] Pipeline done | total_elapsed={total:.2f}s")

    # ── 4. Assemble standard schema matching fusion contract ──────────────────
    return {
        "audio_score": float(scores["s_audio"]),
        "logit_fake":  float(scores["logit_fake"]),
        "logit_real":  float(scores["logit_real"]),
        "transcript":  transcript,
        "wer_confidence_note": "internal sanity-check only, not per-clip WER",
    }
