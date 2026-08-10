"""
Deepfake Classification Module
==============================
Classifies audio clips as real (bonafide) or fake (synthetic/deepfake) using
garystafford/wav2vec2-deepfake-voice-detector from Hugging Face.

Input contract: processed audio as a 16kHz float32 np.ndarray (1D, mono)
                or a file path string. The vad.py strip_silence() output
                is the expected upstream supplier.
Output contract: dict with keys: 's_audio' (float 0-100), 'logit_fake' (float),
                 'logit_real' (float)

Scoring formula:
    S_Audio = Sigmoid(logit_fake - logit_real) * 100

IMPORTANT — label index resolution:
    The fake/real label indices are NOT hardcoded. They are resolved at model
    load time from model.config.id2label and persisted to
    calibration/model_config.json. This file is version-controlled so the
    mapping is auditable and any future model update that changes the label
    order will be caught immediately.

Architecture deviation note:
    File paths are loaded via librosa.load (same as asr.py) to avoid a system
    ffmpeg dependency on Windows. Waveform is truncated to 30s max before
    passing to the processor to avoid out-of-memory issues on long recordings.
"""

import json
import os
from typing import Dict, Union

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ── Module-level model cache ──────────────────────────────────────────────────
_MODEL = None
_PROCESSOR = None
_FAKE_IDX: int = None
_REAL_IDX: int = None

# Path to the persisted label mapping (version-controlled calibration artifact)
_CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__), "calibration", "model_config.json"
)

# Maximum audio duration fed to the model in seconds (wav2vec2 is memory-hungry
# on very long clips; 30s is a safe ceiling that covers nearly all voice samples)
_MAX_SECONDS = 30
_SAMPLE_RATE = 16000
_MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"


# ── Model loading & label resolution ─────────────────────────────────────────

def _load_model():
    """Loads garystafford/wav2vec2-deepfake-voice-detector and resolves label indices.

    Called once on first score_audio() invocation; results are cached in module globals.
    Prints id2label so the mapping is visible in logs at load time.
    Persists the verified mapping into calibration/model_config.json.
    """
    global _MODEL, _PROCESSOR, _FAKE_IDX, _REAL_IDX

    _PROCESSOR = AutoFeatureExtractor.from_pretrained(_MODEL_ID)
    _MODEL = AutoModelForAudioClassification.from_pretrained(_MODEL_ID)
    _MODEL.eval()

    # sanity check for binary classifier - This ensures that if someone accidentally 
    # changes the Hugging Face model to a different checkpoint later, your scoring logic
    # (fake vs real) will fail loudly instead of silently producing incorrect results.
    if _MODEL.config.num_labels != 2:
        raise ValueError(
            f"Expected a binary classifier (2 labels), got {_MODEL.config.num_labels} labels."
        )

    # ── Critical: inspect label mapping; never assume index 0 == real ──────
    id2label = _MODEL.config.id2label
    print(f"[deepfake_model] model.config.id2label = {id2label}")

    # Resolve indices by searching for 'fake'/'real' labels (case-insensitive)
    label2id = {v.lower(): int(k) for k, v in id2label.items()}
    _FAKE_IDX = label2id.get("fake")
    _REAL_IDX = label2id.get("real")

    if _FAKE_IDX is None or _REAL_IDX is None:
        raise ValueError(
            f"[deepfake_model] Could not resolve 'fake'/'real' indices from "
            f"id2label={id2label}. Update label search logic to match actual labels."
        )

    print(f"[deepfake_model] Confirmed -> fake_index={_FAKE_IDX}, real_index={_REAL_IDX}")

    # ── Persist confirmed mapping to calibration/model_config.json ─────────
    calibration_data = {
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
    }
    os.makedirs(os.path.dirname(_CALIBRATION_PATH), exist_ok=True)
    with open(_CALIBRATION_PATH, "w") as f:
        json.dump(calibration_data, f, indent=2)
    print(f"[deepfake_model] Calibration persisted -> {_CALIBRATION_PATH}")


# ── Public API ────────────────────────────────────────────────────────────────

def score_audio(processed_audio: Union[str, np.ndarray]) -> Dict[str, float]:
    """Classifies audio as real or deepfake and returns a 0–100 confidence score.

    Args:
        processed_audio: Either a 16kHz float32 numpy array (1D mono) produced
                         by vad.py, or a file path string. File paths are loaded
                         internally via librosa to avoid an ffmpeg dependency.

    Returns:
        dict: {
            "s_audio": float  — deepfake confidence 0.0 (real) … 100.0 (fake),
            "logit_fake": float  — raw pre-softmax score for the 'fake' class,
            "logit_real": float  — raw pre-softmax score for the 'real' class,
        }
    """
    global _MODEL, _PROCESSOR, _FAKE_IDX, _REAL_IDX

    # Lazy load on first call
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

    # ── 2. Truncate to 30s max to protect memory ──────────────────────────
    max_samples = _MAX_SECONDS * _SAMPLE_RATE
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]

    # Replace NaN or infinite values with 0.0
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

    # ── 3. Feature extraction ─────────────────────────────────────────────
    # AutoFeatureExtractor pads/normalises the waveform into the tensor format
    # wav2vec2 expects: (batch_size, sequence_length)
    inputs = _PROCESSOR(
        waveform,
        sampling_rate=_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    # ── 4. Inference — extract RAW LOGITS, not softmax probabilities ──────
    # We need pre-softmax values so we can compute the sigmoid-of-difference
    # formula: S_Audio = Sigmoid(logit_fake - logit_real) * 100
    # Using softmax probabilities would distort the score because softmax already
    # normalises the outputs to sum to 1, compressing the relative difference.
    with torch.no_grad():
        outputs = _MODEL(**inputs)
    logits = outputs.logits  # shape: (1, num_labels)

    logit_fake = float(logits[0, _FAKE_IDX].item())
    logit_real = float(logits[0, _REAL_IDX].item())

    # ── 5. Scoring formula ────────────────────────────────────────────────
    # Sigmoid squashes the logit difference from (-∞, +∞) into (0, 1),
    # then we scale by 100 to produce a human-readable percentage.
    # Score → 100 means model is highly confident this is a deepfake.
    # Score → 0 means model is highly confident this is a real human voice.
    diff = logit_fake - logit_real
    s_audio = float(torch.sigmoid(torch.tensor(diff)).item()) * 100.0

    return {
        "s_audio": round(s_audio, 4),
        "logit_fake": round(logit_fake, 6),
        "logit_real": round(logit_real, 6),
    }
