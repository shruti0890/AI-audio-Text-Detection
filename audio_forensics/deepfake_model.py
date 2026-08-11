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


# ── Sliding-window scorer ─────────────────────────────────────────────────────

def _score_windows(waveform: np.ndarray) -> List[float]:
    """Score the full waveform using overlapping sliding windows.

    Each window is independently fed through the wav2vec2 classifier.
    Window length and overlap are loaded from model_config.json.

    Args:
        waveform: 1D float32 numpy array at 16kHz.

    Returns:
        List of per-window scores (0–100). Empty list if waveform too short.
    """
    window_samples  = _WINDOW_SECONDS * _SAMPLE_RATE
    overlap_samples = _WINDOW_OVERLAP_SECONDS * _SAMPLE_RATE
    step_samples    = window_samples - overlap_samples

    total_samples = len(waveform)

    # If shorter than one window, score the whole clip as a single window
    if total_samples <= window_samples:
        windows = [waveform]
    else:
        windows = []
        start = 0
        while start + window_samples <= total_samples:
            windows.append(waveform[start: start + window_samples])
            start += step_samples
        # Include a tail window if the last full window doesn't reach the end
        if start < total_samples:
            tail = waveform[start:]
            if len(tail) >= _SAMPLE_RATE:  # at least 1s of audio, otherwise skip
                windows.append(tail)

    scores = []
    for win in windows:
        win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)
        inputs = _PROCESSOR(
            win,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = _MODEL(**inputs)
        logits = outputs.logits  # (1, 2)

        lf = float(logits[0, _FAKE_IDX].item())
        lr = float(logits[0, _REAL_IDX].item())

        # Temperature-scaled sigmoid
        diff_scaled = (lf - lr) / _TEMPERATURE
        s = float(torch.sigmoid(torch.tensor(diff_scaled)).item()) * 100.0
        scores.append(round(s, 4))

    return scores


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

def score_audio(processed_audio: Union[str, np.ndarray]) -> Dict:
    """Classifies audio as real or deepfake and returns a calibrated 0–100 score.

    Uses sliding-window aggregation over the full audio clip (not truncation),
    and applies temperature scaling to logits before sigmoid computation.
    Decision threshold is loaded from model_config.json (not hardcoded).

    Args:
        processed_audio: Either a 16kHz float32 numpy array (1D mono) produced
                         by vad.py, or a file path string.

    Returns:
        dict: {
            "s_audio":            float  — deepfake confidence 0–100,
            "logit_fake":         float  — logit from the highest-risk window,
            "logit_real":         float  — logit from the highest-risk window,
            "decision_threshold": float  — threshold loaded from config (e.g. 62.5),
            "temperature":        float  — temperature factor applied,
            "n_windows":          int    — number of sliding windows processed,
            "window_scores":      list   — per-window scores for debugging,
            "confidence_tiers":   dict   — logit margin tier breakpoints from config,
        }
    """
    global _MODEL, _PROCESSOR, _FAKE_IDX, _REAL_IDX

    if _MODEL is None:
        _load_model()

    # ── 1. Resolve waveform ───────────────────────────────────────────────
    if isinstance(processed_audio, str):
        if not os.path.exists(processed_audio):
            raise FileNotFoundError(f"Audio file not found: {processed_audio}")
        import librosa
        waveform, _ = librosa.load(processed_audio, sr=_SAMPLE_RATE, mono=True)
        waveform = waveform.astype(np.float32)
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

    # ── 2. Sliding-window scoring (replaces hard truncation) ─────────────
    window_scores = _score_windows(waveform)
    n_windows = len(window_scores)

    # ── 3. Aggregate window scores ────────────────────────────────────────
    s_audio = _aggregate_scores(window_scores)

    # ── 4. Retrieve representative logits from the highest-risk window ────
    # Re-score the worst window to get its raw logits for display purposes.
    # (Logits from the max-risk window are the most meaningful to show.)
    if window_scores:
        worst_window_idx = int(np.argmax(window_scores))
        window_samples  = _WINDOW_SECONDS * _SAMPLE_RATE
        overlap_samples = _WINDOW_OVERLAP_SECONDS * _SAMPLE_RATE
        step_samples    = window_samples - overlap_samples

        total_samples = len(waveform)
        if total_samples <= window_samples:
            worst_win = waveform
        else:
            start = worst_window_idx * step_samples
            end   = start + window_samples
            worst_win = waveform[start: min(end, total_samples)]

        worst_win = np.nan_to_num(worst_win, nan=0.0, posinf=0.0, neginf=0.0)
        inputs = _PROCESSOR(
            worst_win,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = _MODEL(**inputs)
        logits = outputs.logits
        logit_fake = round(float(logits[0, _FAKE_IDX].item()), 6)
        logit_real = round(float(logits[0, _REAL_IDX].item()), 6)
    else:
        logit_fake = 0.0
        logit_real = 0.0

    print(
        f"[deepfake_model] n_windows={n_windows}  scores={window_scores}  "
        f"aggregated={s_audio:.4f}  threshold={_DECISION_THRESHOLD}%  T={_TEMPERATURE}"
    )

    return {
        "s_audio":            round(s_audio, 4),
        "logit_fake":         logit_fake,
        "logit_real":         logit_real,
        "decision_threshold": _DECISION_THRESHOLD,
        "temperature":        _TEMPERATURE,
        "n_windows":          n_windows,
        "window_scores":      window_scores,
        "confidence_tiers":   _CONFIDENCE_TIERS,
    }
