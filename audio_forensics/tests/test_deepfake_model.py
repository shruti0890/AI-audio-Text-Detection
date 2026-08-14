"""Tests for audio_forensics/deepfake_model.py — Phase 2.4"""

import json
import os

import numpy as np
import pytest

from audio_forensics.deepfake_model import score_audio

SAMPLE_CLIPS = os.path.join(os.path.dirname(__file__), "..", "sample_clips")
SPEECH_CLIP = os.path.join(SAMPLE_CLIPS, "speech_sample.wav")
CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__), "..", "calibration", "model_config.json"
)


# ── Schema / return-type tests ────────────────────────────────────────────────

def test_score_audio_returns_correct_keys():
    """score_audio() must return a dict with exactly the three required keys."""
    dummy = np.zeros(16000, dtype=np.float32)
    res = score_audio(dummy)
    assert isinstance(res, dict), "Return value must be a dict"
    assert "s_audio" in res
    assert "logit_fake" in res
    assert "logit_real" in res


def test_score_audio_types():
    """All three return values must be Python floats."""
    dummy = np.zeros(16000, dtype=np.float32)
    res = score_audio(dummy)
    assert isinstance(res["s_audio"], float), "s_audio must be a float"
    assert isinstance(res["logit_fake"], float), "logit_fake must be a float"
    assert isinstance(res["logit_real"], float), "logit_real must be a float"


def test_s_audio_range():
    """s_audio must be in [0.0, 100.0] — it's a Sigmoid-scaled percentage."""
    dummy = np.zeros(16000, dtype=np.float32)
    res = score_audio(dummy)
    assert 0.0 <= res["s_audio"] <= 100.0, (
        f"s_audio={res['s_audio']} is out of [0, 100] range"
    )


# ── Formula verification ──────────────────────────────────────────────────────

def test_sigmoid_formula_consistency():
    """Verify s_audio == Sigmoid((logit_fake - logit_real) / T) * 100."""
    import torch
    dummy = np.zeros(16000, dtype=np.float32)
    res = score_audio(dummy)
    diff = res["logit_fake"] - res["logit_real"]
    temp = res.get("temperature", 1.15)
    expected = float(torch.sigmoid(torch.tensor(diff / temp)).item()) * 100.0
    assert abs(res["s_audio"] - expected) < 0.001, (
        f"Formula mismatch: got {res['s_audio']}, expected {expected:.4f}"
    )


# ── File path input ───────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(SPEECH_CLIP),
    reason="speech_sample.wav not found in sample_clips/"
)
def test_score_audio_from_filepath():
    """score_audio() must accept a file path and return a valid result."""
    res = score_audio(SPEECH_CLIP)
    assert isinstance(res, dict)
    assert 0.0 <= res["s_audio"] <= 100.0


# ── Calibration persistence ───────────────────────────────────────────────────

def test_calibration_json_written_after_load():
    """After model load, calibration/model_config.json must exist and have 'verified' status."""
    # Trigger model load by calling score_audio
    score_audio(np.zeros(16000, dtype=np.float32))
    assert os.path.exists(CALIBRATION_PATH), (
        "calibration/model_config.json was not written after model load"
    )
    with open(CALIBRATION_PATH) as f:
        cfg = json.load(f)
    assert cfg.get("status") == "verified", (
        f"Expected status='verified', got status='{cfg.get('status')}'"
    )
    assert isinstance(cfg.get("fake_index"), int), "fake_index must be an int"
    assert isinstance(cfg.get("real_index"), int), "real_index must be an int"
    assert cfg["fake_index"] != cfg["real_index"], (
        "fake_index and real_index must differ"
    )


def test_label_indices_not_hardcoded():
    """Label indices must be derived from model config, not assumed."""
    assert os.path.exists(CALIBRATION_PATH), "Run test_calibration_json_written_after_load first"
    with open(CALIBRATION_PATH) as f:
        cfg = json.load(f)
    id2label = cfg.get("id2label", {})
    fake_label = id2label.get(str(cfg["fake_index"]), "").lower()
    real_label = id2label.get(str(cfg["real_index"]), "").lower()
    assert fake_label == "fake", f"Index {cfg['fake_index']} should map to 'fake', got '{fake_label}'"
    assert real_label == "real", f"Index {cfg['real_index']} should map to 'real', got '{real_label}'"
