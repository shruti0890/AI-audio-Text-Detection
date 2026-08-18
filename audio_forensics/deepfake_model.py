"""
Deepfake Classification Module
==============================
Classifies audio clips as real (bonafide) or fake (synthetic/deepfake) using
garystafford/wav2vec2-deepfake-voice-detector from Hugging Face.

Input contract:  processed audio as a 16kHz float32 np.ndarray (1D, mono)
                 or a file path string. The vad.py strip_silence() output
                 is the expected upstream supplier.

Output contract: dict with keys:
    's_audio'            (float 0-100)    deepfake confidence score
    'logit_fake'         (float)          raw pre-softmax logit for 'fake'
    'logit_real'         (float)          raw pre-softmax logit for 'real'
    'decision_threshold' (float)          threshold loaded from model_config.json
    'temperature'        (float)          temperature factor applied to logits
    'n_windows'          (int)            number of sliding windows processed
    'window_scores'      (list[float])    per-window scores (for debugging)

Scoring formula (with temperature scaling):
    diff_scaled = (logit_fake - logit_real) / T
    S_Audio = Sigmoid(diff_scaled) * 100

where T = calibration_parameters.temperature_scaling from model_config.json.

IMPORTANT — label index resolution:
    The fake/real label indices are NOT hardcoded. They are resolved at model
    load time from model.config.id2label and persisted to
    calibration/model_config.json. This file is version-controlled so the
    mapping is auditable and any future model update that changes the label
    order will be caught immediately.

Sliding-window aggregation:
    Instead of truncating to the first N seconds, the full waveform is
    processed in overlapping windows (window_seconds, window_overlap_seconds
    from model_config.json). Each window produces an independent score.
    Final score = max(window_scores) under 'max_risk' aggregation strategy
    (conservative: flag the clip as fake if ANY window looks synthetic).
    This catches deepfake artifacts that occur past the first 30s.

Architecture deviation note:
    File paths are loaded via librosa.load (same as asr.py) to avoid a system
    ffmpeg dependency on Windows.
"""

import json
import os
from typing import Dict, List, Union

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ── Module-level model cache ──────────────────────────────────────────────────
_MODEL = None
_PROCESSOR = None
_FAKE_IDX: int = None
_REAL_IDX: int = None

# Calibration parameters — loaded from model_config.json at model-load time
_DECISION_THRESHOLD: float = 62.5   # placeholder; overwritten by config
_TEMPERATURE: float = 1.15          # placeholder; overwritten by config
_WINDOW_SECONDS: int = 5            # sliding window length
_WINDOW_OVERLAP_SECONDS: int = 1    # overlap between windows
_AGGREGATION: str = "max_risk"      # aggregation strategy
_CONFIDENCE_TIERS: dict = {         # logit margin breakpoints
    "extreme_logit_margin": 3.0,
    "high_logit_margin": 1.5,
    "moderate_logit_margin": 0.5,
}

# Path to the persisted label mapping (version-controlled calibration artifact)
_CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__), "calibration", "model_config.json"
)

_SAMPLE_RATE = 16000
_MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"


# ── Calibration helpers ───────────────────────────────────────────────────────

def _load_calibration_params() -> dict:
    """Read calibration_parameters from model_config.json.
    Returns defaults if the key is absent (first-run before _load_model writes).
    """
    global _DECISION_THRESHOLD, _TEMPERATURE, _WINDOW_SECONDS
    global _WINDOW_OVERLAP_SECONDS, _AGGREGATION, _CONFIDENCE_TIERS

    if not os.path.exists(_CALIBRATION_PATH):
        return {}

    try:
        with open(_CALIBRATION_PATH, "r") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    params = cfg.get("calibration_parameters", {})
    if params:
        _DECISION_THRESHOLD = float(params.get("decision_threshold_pct", _DECISION_THRESHOLD))
        _TEMPERATURE        = float(params.get("temperature_scaling", _TEMPERATURE))
        _WINDOW_SECONDS     = int(params.get("window_seconds", _WINDOW_SECONDS))
        _WINDOW_OVERLAP_SECONDS = int(params.get("window_overlap_seconds", _WINDOW_OVERLAP_SECONDS))
        _AGGREGATION        = str(params.get("aggregation_strategy", _AGGREGATION))
        _CONFIDENCE_TIERS   = params.get("confidence_tiers", _CONFIDENCE_TIERS)
        print(
            f"[deepfake_model] Calibration loaded — threshold={_DECISION_THRESHOLD}%  "
            f"temperature={_TEMPERATURE}  window={_WINDOW_SECONDS}s  "
            f"overlap={_WINDOW_OVERLAP_SECONDS}s  agg={_AGGREGATION}"
        )
    return params


def _merge_write_config(new_label_data: dict) -> None:
    """Write label-index data back to model_config.json WITHOUT clobbering
    the calibration_parameters block that may have been manually edited.
    """
    existing = {}
    if os.path.exists(_CALIBRATION_PATH):
        try:
            with open(_CALIBRATION_PATH, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    # Merge: label data overwrites label keys; calibration block is preserved
    merged = {**existing, **new_label_data}
    if "calibration_parameters" in existing:
        merged["calibration_parameters"] = existing["calibration_parameters"]

    os.makedirs(os.path.dirname(_CALIBRATION_PATH), exist_ok=True)
    with open(_CALIBRATION_PATH, "w") as f:
        json.dump(merged, f, indent=2)


# ── Model loading & label resolution ─────────────────────────────────────────

def _load_model():
    """Loads wav2vec2-deepfake-voice-detector and resolves label indices.

    Called once on first score_audio() invocation; results are cached in
    module globals. Also loads calibration parameters from model_config.json.
    """
    global _MODEL, _PROCESSOR, _FAKE_IDX, _REAL_IDX

    # Load calibration params FIRST (so threshold is ready before scoring)
    _load_calibration_params()

    try:
        _PROCESSOR = AutoFeatureExtractor.from_pretrained(_MODEL_ID, local_files_only=True)
        _MODEL = AutoModelForAudioClassification.from_pretrained(_MODEL_ID, local_files_only=True)
    except Exception:
        _PROCESSOR = AutoFeatureExtractor.from_pretrained(_MODEL_ID)
        _MODEL = AutoModelForAudioClassification.from_pretrained(_MODEL_ID)
    _MODEL.eval()

    if _MODEL.config.num_labels != 2:
        raise ValueError(
            f"Expected a binary classifier (2 labels), got {_MODEL.config.num_labels} labels."
        )

    id2label = _MODEL.config.id2label
    print(f"[deepfake_model] model.config.id2label = {id2label}")

    label2id = {v.lower(): int(k) for k, v in id2label.items()}
    _FAKE_IDX = label2id.get("fake")
    _REAL_IDX = label2id.get("real")

    if _FAKE_IDX is None or _REAL_IDX is None:
        raise ValueError(
            f"[deepfake_model] Could not resolve 'fake'/'real' indices from "
            f"id2label={id2label}. Update label search logic to match actual labels."
        )

    print(f"[deepfake_model] Confirmed -> fake_index={_FAKE_IDX}, real_index={_REAL_IDX}")

    # Persist label mapping back, preserving calibration_parameters block
    _merge_write_config({
        "model_name": _MODEL_ID,
        "id2label": {str(k): v for k, v in id2label.items()},
        "fake_index": _FAKE_IDX,
        "real_index": _REAL_IDX,
        "status": "verified",
        "note": (
            "Label indices confirmed programmatically from model.config.id2label "
            "at model load time. Do NOT hardcode these values — always resolve from "
            "model config so a future checkpoint update does not silently break scoring."
        ),
    })
    print(f"[deepfake_model] Config persisted (calibration_parameters preserved) -> {_CALIBRATION_PATH}")


import gc
from typing import Callable, Dict, List, Optional, Tuple, Union

# ── Sliding-window generator & memory-bounded scorer ──────────────────────────

def _generate_window_slices(total_samples: int) -> List[Tuple[int, int]]:
    """Generate (start, end) sample index slices for sliding windows.

    Config parameters:
        window_seconds = 5
        window_overlap_seconds = 1
        step_seconds = 4
        tail window kept if >= 1 second
    """
    window_samples  = _WINDOW_SECONDS * _SAMPLE_RATE
    overlap_samples = _WINDOW_OVERLAP_SECONDS * _SAMPLE_RATE
    step_samples    = window_samples - overlap_samples

    if total_samples <= window_samples:
        return [(0, total_samples)]

    slices = []
    start = 0
    while start + window_samples <= total_samples:
        slices.append((start, start + window_samples))
        start += step_samples

    # Include tail segment if >= 1s (16,000 samples)
    if start < total_samples:
        tail_len = total_samples - start
        if tail_len >= _SAMPLE_RATE:
            slices.append((start, total_samples))

    return slices


def _score_windows_bounded(
    waveform: np.ndarray,
    batch_size: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, float]]:
    """Score the waveform using memory-bounded small inference batches (default batch_size=1).

    Generates window slices lazily, feeds at most `batch_size` windows to wav2vec2
    at a time under torch.inference_mode(), extracts numeric scores immediately,
    and deletes temporary tensors after each step to keep peak memory minimal.

    Args:
        waveform: 1D float32 numpy array at 16kHz.
        batch_size: Max number of windows processed simultaneously (default: 1).
        progress_callback: Optional callback func(current_window_idx, total_windows).

    Returns:
        List of dicts: [{'score': float, 'logit_fake': float, 'logit_real': float}]
    """
    total_samples = len(waveform)
    slices = _generate_window_slices(total_samples)
    total_windows = len(slices)

    if total_windows == 0:
        return []

    results = []
    effective_batch_size = max(1, batch_size)

    for b_start in range(0, total_windows, effective_batch_size):
        b_slices = slices[b_start : b_start + effective_batch_size]
        batch_windows = [
            np.nan_to_num(waveform[s_start:s_end], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            for s_start, s_end in b_slices
        ]

        # Prepare input tensors for the small batch
        inputs = _PROCESSOR(
            batch_windows,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        with torch.inference_mode():
            outputs = _MODEL(**inputs)
            logits = outputs.logits  # Shape: (batch_len, 2)

        for i in range(len(b_slices)):
            lf = float(logits[i, _FAKE_IDX].item())
            lr = float(logits[i, _REAL_IDX].item())

            # Temperature-scaled sigmoid formula: S_i = Sigmoid((L_fake - L_real) / T) * 100
            diff_scaled = (lf - lr) / _TEMPERATURE
            s = float(torch.sigmoid(torch.tensor(diff_scaled)).item()) * 100.0

            results.append({
                "score": round(s, 4),
                "logit_fake": round(lf, 6),
                "logit_real": round(lr, 6),
            })

            cur_idx = len(results)
            if progress_callback:
                try:
                    progress_callback(cur_idx, total_windows)
                except Exception:
                    pass

        # Explicitly release temporary tensors for this batch
        del inputs, outputs, logits, batch_windows

    return results


def _aggregate_scores(scores: List[float]) -> float:
    """Aggregate per-window scores into a single clip-level score.

    Strategies:
        max_risk   — max(scores): conservative, flags clip if ANY window looks fake
        mean       — mean(scores): balanced view
        p90        — 90th percentile: robust to a single noisy window
    """
    if not scores:
        return 50.0  # silence / empty fallback

    strategy = _AGGREGATION.lower()
    if strategy == "max_risk":
        return float(max(scores))
    elif strategy == "mean":
        return float(np.mean(scores))
    elif strategy == "p90":
        return float(np.percentile(scores, 90))
    else:
        return float(max(scores))  # default to max_risk if unknown strategy


# ── Public API ────────────────────────────────────────────────────────────────

def score_audio(
    processed_audio: Union[str, np.ndarray],
    batch_size: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """Classifies audio as real or deepfake and returns a calibrated 0–100 score.

    Uses memory-bounded sliding-window aggregation over the full audio clip
    with default batch_size=1 (preventing CPU OOM crashes on long files),
    and applies temperature scaling to logits before sigmoid computation.
    Decision threshold is loaded from model_config.json.

    Args:
        processed_audio: Either a 16kHz float32 numpy array (1D mono) produced
                         by vad.py, or a file path string.
        batch_size: Max windows processed per model forward pass (default: 1).
        progress_callback: Optional callable(current_window, total_windows).

    Returns:
        dict: {
            "s_audio":               float — deepfake confidence 0–100 (max_risk),
            "max_score":             float — highest window score,
            "mean_score":            float — average across windows,
            "median_score":          float — median across windows,
            "logit_fake":            float — logit from the highest-risk window,
            "logit_real":            float — logit from the highest-risk window,
            "decision_threshold":    float — threshold loaded from config (e.g. 56.0),
            "temperature":           float — temperature factor applied,
            "n_windows":             int   — number of sliding windows processed,
            "windows_analyzed":      int   — alias for n_windows,
            "audio_duration_seconds": float — total audio duration in seconds,
            "window_scores":         list  — per-window scores,
            "confidence_tiers":      dict  — logit margin tier breakpoints,
            "inference_batch_size":  int   — batch size used during inference (1),
            "window_size_seconds":   int   — window length (5),
            "window_overlap_seconds": int  — overlap duration (1),
            "window_step_seconds":   int   — step size (4),
        }
    """
    global _MODEL, _PROCESSOR, _FAKE_IDX, _REAL_IDX

    if _MODEL is None:
        _load_model()

    # ── 1. Resolve waveform ───────────────────────────────────────────────
    if isinstance(processed_audio, str):
        from .audio_loader import load_and_normalize_audio
        waveform, _ = load_and_normalize_audio(processed_audio, target_sr=_SAMPLE_RATE)
    elif isinstance(processed_audio, np.ndarray):
        waveform = processed_audio.astype(np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
    elif isinstance(processed_audio, torch.Tensor):
        waveform = processed_audio.detach().cpu().numpy().astype(np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
    else:
        raise ValueError(f"Unsupported audio input type: {type(processed_audio)}")

    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    duration_sec = round(len(waveform) / _SAMPLE_RATE, 2)

    # ── 2. Memory-bounded sliding-window scoring ─────────────────────────
    window_results = _score_windows_bounded(
        waveform,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
    window_scores = [w["score"] for w in window_results]
    n_windows = len(window_scores)

    # ── 3. Aggregate window scores (max_risk preserved) ───────────────────
    s_audio = _aggregate_scores(window_scores)

    # Additional summary statistics
    mean_score = round(float(np.mean(window_scores)), 4) if window_scores else 50.0
    median_score = round(float(np.median(window_scores)), 4) if window_scores else 50.0

    # ── 4. Retrieve representative logits from the highest-risk window ────
    if window_results:
        worst_window_idx = int(np.argmax(window_scores))
        logit_fake = window_results[worst_window_idx]["logit_fake"]
        logit_real = window_results[worst_window_idx]["logit_real"]
    else:
        logit_fake = 0.0
        logit_real = 0.0

    print(
        f"[deepfake_model] n_windows={n_windows} (duration={duration_sec}s, batch_size={batch_size}) "
        f"max={s_audio:.2f}% mean={mean_score:.2f}% median={median_score:.2f}% "
        f"threshold={_DECISION_THRESHOLD}% T={_TEMPERATURE}"
    )

    return {
        "s_audio":               round(s_audio, 4),
        "max_score":             round(s_audio, 4),
        "mean_score":            mean_score,
        "median_score":          median_score,
        "logit_fake":            logit_fake,
        "logit_real":            logit_real,
        "decision_threshold":    _DECISION_THRESHOLD,
        "temperature":           _TEMPERATURE,
        "n_windows":             n_windows,
        "windows_analyzed":      n_windows,
        "audio_duration_seconds": duration_sec,
        "window_scores":         window_scores,
        "confidence_tiers":      _CONFIDENCE_TIERS,
        "inference_batch_size":  batch_size,
        "window_size_seconds":   _WINDOW_SECONDS,
        "window_overlap_seconds": _WINDOW_OVERLAP_SECONDS,
        "window_step_seconds":   _WINDOW_SECONDS - _WINDOW_OVERLAP_SECONDS,
    }

