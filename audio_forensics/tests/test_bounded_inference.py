"""Tests for audio_forensics memory-bounded sliding window and normalization."""

import os
import numpy as np
import pytest

from audio_forensics.audio_loader import load_and_normalize_audio, SUPPORTED_AUDIO_EXTENSIONS
from audio_forensics.deepfake_model import score_audio, _generate_window_slices


def test_supported_extensions():
    assert ".wav" in SUPPORTED_AUDIO_EXTENSIONS
    assert ".mp3" in SUPPORTED_AUDIO_EXTENSIONS
    assert ".flac" in SUPPORTED_AUDIO_EXTENSIONS
    assert ".ogg" in SUPPORTED_AUDIO_EXTENSIONS
    assert ".m4a" in SUPPORTED_AUDIO_EXTENSIONS
    assert ".aac" in SUPPORTED_AUDIO_EXTENSIONS


def test_load_and_normalize_numpy_array():
    sr = 16000
    # 2 seconds stereo
    stereo = np.random.randn(2, sr * 2).astype(np.float32)
    norm_wave, meta = load_and_normalize_audio(stereo, target_sr=sr)

    assert isinstance(norm_wave, np.ndarray)
    assert norm_wave.ndim == 1
    assert norm_wave.dtype == np.float32
    assert len(norm_wave) == sr * 2
    assert meta["normalized_sample_rate"] == 16000
    assert meta["normalized_channels"] == 1
    assert meta["original_channels"] == 2
    assert meta["normalized_duration_seconds"] == 2.0


def test_generate_window_slices_short():
    sr = 16000
    # 3 seconds audio (<= 5s window) -> exactly 1 window
    slices = _generate_window_slices(3 * sr)
    assert len(slices) == 1
    assert slices[0] == (0, 3 * sr)


def test_generate_window_slices_long():
    sr = 16000
    # 6 minutes 10 seconds = 370 seconds
    # Windows: (0, 5s), (4s, 9s), (8s, 13s), ...
    # Step = 4s.
    # 370 - 5 = 365. 365 // 4 = 91 full steps -> 92 windows.
    # start = 92 * 4 = 368s. Remaining tail = 370 - 368 = 2s (>= 1s) -> 1 tail window.
    # Total = 93 windows.
    total_samples = 370 * sr
    slices = _generate_window_slices(total_samples)
    assert len(slices) == 93
    # Check first few
    assert slices[0] == (0, 5 * sr)
    assert slices[1] == (4 * sr, 9 * sr)
    assert slices[2] == (8 * sr, 13 * sr)
    # Check tail
    assert slices[-1] == (368 * sr, 370 * sr)


def test_score_audio_bounded_memory():
    sr = 16000
    # 12 seconds audio: (0-5s), (4-9s), (8-12s, tail >= 1s) -> 3 windows
    dummy_waveform = np.zeros(12 * sr, dtype=np.float32)

    progress_counts = []
    def callback(cur, total):
        progress_counts.append((cur, total))

    res = score_audio(dummy_waveform, batch_size=1, progress_callback=callback)

    assert isinstance(res, dict)
    assert res["n_windows"] == 3
    assert res["windows_analyzed"] == 3
    assert res["inference_batch_size"] == 1
    assert "max_score" in res
    assert "mean_score" in res
    assert "median_score" in res
    assert len(res["window_scores"]) == 3
    assert len(progress_counts) == 3
    assert progress_counts[-1] == (3, 3)
