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
from typing import Callable, Dict, Any, Optional

from .audio_loader import load_and_normalize_audio
from .vad import strip_silence
from .asr import transcribe
from .deepfake_model import score_audio


def analyze_audio(
    audio_path: str,
    batch_size: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Runs the full audio forensics pipeline on an input file.

    Flow:
        load_and_normalize_audio(audio_path)
            → strip_silence(waveform)
            → transcribe(processed_audio)
            → score_audio(processed_audio, batch_size=1, progress_callback)

    Args:
        audio_path (str): Path to input audio clip (.wav, .mp3, .flac, .ogg, .m4a, .aac).
        batch_size (int): Inference batch size for sliding-window evaluation (default: 1).
        progress_callback: Optional callable(current_window, total_windows) for live progress.

    Returns:
        Dict[str, Any]: Audio forensic evaluation results formatted for fusion consumption.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"[pipeline] Audio file not found: {audio_path}")

    t_start = time.perf_counter()
    print(f"[pipeline] Starting analysis -> {audio_path}")

    # ── 0. Decode & Normalize ONCE ───────────────────────────────────────────
    raw_waveform, audio_meta = load_and_normalize_audio(audio_path, target_sr=16000)

    # ── 1. Voice Activity Detection (Silence Stripping) ───────────────────────
    t0 = time.perf_counter()
    processed_audio = strip_silence(raw_waveform)
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
    scores = score_audio(
        processed_audio,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
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
        "audio_score":            float(scores["s_audio"]),
        "max_score":              float(scores.get("max_score", scores["s_audio"])),
        "mean_score":             float(scores.get("mean_score", scores["s_audio"])),
        "median_score":           float(scores.get("median_score", scores["s_audio"])),
        "logit_fake":             float(scores["logit_fake"]),
        "logit_real":             float(scores["logit_real"]),
        "transcript":             transcript,
        "wer_confidence_note":    "internal sanity-check only, not per-clip WER",
        "decision_threshold_pct": float(scores["decision_threshold"]),
        "temperature":            float(scores["temperature"]),
        "n_windows":              int(scores["n_windows"]),
        "windows_analyzed":       int(scores.get("windows_analyzed", scores["n_windows"])),
        "audio_duration_seconds": float(scores.get("audio_duration_seconds", audio_meta["normalized_duration_seconds"])),
        "window_scores":          list(scores["window_scores"]),
        "confidence_tiers":       dict(scores["confidence_tiers"]),
        "inference_batch_size":   int(scores.get("inference_batch_size", batch_size)),
        "window_size_seconds":    int(scores.get("window_size_seconds", 5)),
        "window_overlap_seconds": int(scores.get("window_overlap_seconds", 1)),
        "window_step_seconds":    int(scores.get("window_step_seconds", 4)),
        "audio_metadata":         audio_meta,
        "vad_time":               float(vad_time),
        "asr_time":               float(asr_time),
        "deepfake_time":          float(deepfake_time),
        "total_time":             float(total),
    }
