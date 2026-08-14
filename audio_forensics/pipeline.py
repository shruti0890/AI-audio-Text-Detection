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
        "audio_score":          float,  # deepfake confidence 0.0 (real) … 100.0 (fake)
        "logit_fake":           float,  # raw pre-softmax logit for the 'fake' class
        "logit_real":           float,  # raw pre-softmax logit for the 'real' class
        "transcript":           str,    # spoken text from Whisper-Tiny (used by cross-modal check)
        "wer_confidence_note":  str,    # fixed human-readable caveat for fusion consumers
        "decision_threshold_pct": float, # threshold loaded from model_config.json (e.g. 62.5)
        "temperature":          float,  # temperature scaling factor applied to logits
        "n_windows":            int,    # number of sliding windows processed
        "window_scores":        list,   # per-window scores for debugging / audit
        "confidence_tiers":     dict,   # logit margin tier breakpoints from config
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

    Decision threshold and temperature scaling are loaded dynamically from
    audio_forensics/calibration/model_config.json (calibration_parameters block).
    These are currently placeholder values derived from ASVspoof 2021 benchmarks;
    they will be replaced after empirical calibration per AUDIO_CALIBRATION_PROTOCOL.md.

    Args:
        audio_path (str): Path to input audio clip (.wav, .mp3, etc.).
                          Must be an existing file; FileNotFoundError is raised otherwise.

    Returns:
        Dict[str, Any]: Audio forensic evaluation results formatted for fusion consumption.
            Key names are contractually fixed and must not be changed.

            Example output:
                {
                    "audio_score":           81.2,
                    "logit_fake":            2.14,
                    "logit_real":           -0.87,
                    "transcript":            "text transcribed from audio...",
                    "wer_confidence_note":   "internal sanity-check only, not per-clip WER",
                    "decision_threshold_pct": 62.5,
                    "temperature":           1.15,
                    "n_windows":             6,
                    "window_scores":         [79.1, 82.4, 81.2, 84.0, 80.5, 79.8],
                    "confidence_tiers":      {"extreme_logit_margin": 3.0, ...},
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
    vad_time = time.perf_counter() - t0
    print(f"[pipeline] VAD complete  | samples={len(processed_audio):,} "
          f"| duration={len(processed_audio)/16000:.2f}s "
          f"| elapsed={vad_time:.2f}s")

    # ── 2. Speech-to-Text Transcription (Whisper-Tiny) ────────────────────────
    t0 = time.perf_counter()
    transcript = transcribe(processed_audio)
    asr_time = time.perf_counter() - t0
    print(f"[pipeline] ASR complete  | chars={len(transcript)} "
          f"| elapsed={asr_time:.2f}s")
    print(f"[pipeline] Transcript    | \"{transcript[:120]}{'...' if len(transcript) > 120 else ''}\"")

    # ── 3. Deepfake Classification (wav2vec2-deepfake-voice-detector) ─────────
    t0 = time.perf_counter()
    scores = score_audio(processed_audio)
    deepfake_time = time.perf_counter() - t0
    print(f"[pipeline] Deepfake done | s_audio={scores['s_audio']:.4f} "
          f"threshold={scores['decision_threshold']}% "
          f"T={scores['temperature']} "
          f"n_windows={scores['n_windows']} "
          f"logit_fake={scores['logit_fake']:.6f} logit_real={scores['logit_real']:.6f} "
          f"| elapsed={deepfake_time:.2f}s")

    total = time.perf_counter() - t_start
    print(f"[pipeline] Pipeline done | total_elapsed={total:.2f}s")

    # ── 4. Assemble standard schema matching fusion contract ──────────────────
    return {
        "audio_score":           float(scores["s_audio"]),
        "logit_fake":            float(scores["logit_fake"]),
        "logit_real":            float(scores["logit_real"]),
        "transcript":            transcript,
        "wer_confidence_note":   "internal sanity-check only, not per-clip WER",
        "decision_threshold_pct": float(scores["decision_threshold"]),
        "temperature":           float(scores["temperature"]),
        "n_windows":             int(scores["n_windows"]),
        "window_scores":         list(scores["window_scores"]),
        "confidence_tiers":      dict(scores["confidence_tiers"]),
        "vad_time":              float(vad_time),
        "asr_time":              float(asr_time),
        "deepfake_time":         float(deepfake_time),
        "total_time":            float(total),
    }
