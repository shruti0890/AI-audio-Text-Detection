"""
test_audio_pipeline.py
======================
Phase 2.6 — Testing & Validation suite for audio_forensics/pipeline.py.

Tests cover:
    1. Full end-to-end pipeline execution on synthetic tone clip (schema + types)
    2. Value range validation for audio_score (0.0 to 100.0)
    3. Transcript type enforcement (plain str)
    4. FileNotFoundError raised for nonexistent audio path
    5. Fixed wer_confidence_note contract string check
    6. [Phase 2.6] Real speech audio clip test (speech_sample.wav)
    7. [Phase 2.6] Non-speech synthetic tone clip test
    8. [Phase 2.6] Heavy silence audio clip test (80% silence + 20% tone signal)
"""

import os
import pytest
import numpy as np
import soundfile as sf
from audio_forensics.pipeline import analyze_audio

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_clips")
SAMPLE_CLIP_PATH = os.path.join(SAMPLE_DIR, "test_clip_with_silence.wav")
SPEECH_CLIP_PATH = os.path.join(SAMPLE_DIR, "speech_sample.wav")
HEAVY_SILENCE_CLIP_PATH = os.path.join(SAMPLE_DIR, "heavy_silence_clip.wav")


def setup_module():
    """Ensure sample audio clips exist before running pipeline tests."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    # 1. Standard test clip (1.5s silence + 2s tone + 1.5s silence)
    if not os.path.exists(SAMPLE_CLIP_PATH):
        sr = 16000
        silence = np.zeros(int(sr * 1.5), dtype=np.float32)
        t = np.linspace(0, 2, sr * 2, False)
        signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio = np.concatenate([silence, signal, silence])
        sf.write(SAMPLE_CLIP_PATH, audio, sr)

    # 2. Heavy silence clip (4s silence + 1s tone + 4s silence -> 88% silence)
    if not os.path.exists(HEAVY_SILENCE_CLIP_PATH):
        sr = 16000
        silence_lead = np.zeros(int(sr * 4.0), dtype=np.float32)
        silence_trail = np.zeros(int(sr * 4.0), dtype=np.float32)
        t = np.linspace(0, 1, sr * 1, False)
        signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio = np.concatenate([silence_lead, signal, silence_trail])
        sf.write(HEAVY_SILENCE_CLIP_PATH, audio, sr)


def test_analyze_audio_pipeline():
    """Full pipeline returns a dict with all five required keys and correct types."""
    assert os.path.exists(SAMPLE_CLIP_PATH)
    res = analyze_audio(SAMPLE_CLIP_PATH)
    assert isinstance(res, dict)
    required_keys = {
        "audio_score",
        "logit_fake",
        "logit_real",
        "transcript",
        "wer_confidence_note",
    }
    assert required_keys.issubset(res.keys()), (
        f"Missing keys: {required_keys - res.keys()}"
    )
    assert isinstance(res["audio_score"], float)
    assert isinstance(res["logit_fake"], float)
    assert isinstance(res["logit_real"], float)
    assert isinstance(res["transcript"], str)
    assert isinstance(res["wer_confidence_note"], str)


def test_audio_score_range():
    """audio_score must be in [0.0, 100.0] (Sigmoid * 100 guarantee)."""
    res = analyze_audio(SAMPLE_CLIP_PATH)
    assert 0.0 <= res["audio_score"] <= 100.0, (
        f"audio_score {res['audio_score']} is out of [0, 100] range"
    )


def test_transcript_is_string():
    """Transcript must be a plain string (not a dict or list)."""
    res = analyze_audio(SAMPLE_CLIP_PATH)
    assert isinstance(res["transcript"], str), (
        f"Expected str, got {type(res['transcript'])}"
    )


def test_missing_file_raises_error():
    """FileNotFoundError must be raised for a nonexistent audio path."""
    with pytest.raises(FileNotFoundError):
        analyze_audio("/nonexistent/path/to/audio.wav")


def test_wer_confidence_note_is_fixed_string():
    """wer_confidence_note must match the fixed contract string."""
    res = analyze_audio(SAMPLE_CLIP_PATH)
    assert res["wer_confidence_note"] == "internal sanity-check only, not per-clip WER"


# ── Phase 2.6 Scenario Validation Tests ──────────────────────────────────────

def test_human_speech_clip():
    """Phase 2.6 Scenario 1: Test pipeline on spoken speech clip if present."""
    if not os.path.exists(SPEECH_CLIP_PATH):
        pytest.skip(f"Speech sample not found at {SPEECH_CLIP_PATH}")
    res = analyze_audio(SPEECH_CLIP_PATH)
    assert isinstance(res["audio_score"], float)
    assert 0.0 <= res["audio_score"] <= 100.0
    assert isinstance(res["transcript"], str)
    assert len(res["transcript"]) > 0


def test_synthetic_tone_clip():
    """Phase 2.6 Scenario 2: Test pipeline on non-speech synthetic tone clip."""
    res = analyze_audio(SAMPLE_CLIP_PATH)
    assert isinstance(res["audio_score"], float)
    assert "logit_fake" in res
    assert "logit_real" in res


def test_silence_heavy_clip():
    """Phase 2.6 Scenario 3: Test pipeline on 88% silence-heavy clip."""
    assert os.path.exists(HEAVY_SILENCE_CLIP_PATH)
    res = analyze_audio(HEAVY_SILENCE_CLIP_PATH)
    assert isinstance(res["audio_score"], float)
    assert 0.0 <= res["audio_score"] <= 100.0
    assert isinstance(res["transcript"], str)
