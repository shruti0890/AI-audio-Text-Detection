import os
import numpy as np
import soundfile as sf
from audio_forensics.vad import strip_silence

SAMPLE_CLIP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "sample_clips", "test_clip_with_silence.wav"
)


def setup_module():
    """Ensure a sample audio clip with silence exists before running tests."""
    os.makedirs(os.path.dirname(SAMPLE_CLIP_PATH), exist_ok=True)
    if not os.path.exists(SAMPLE_CLIP_PATH):
        sr = 16000
        # 1.5 seconds silence, 2 seconds active tone/signal, 1.5 seconds silence
        silence = np.zeros(int(sr * 1.5), dtype=np.float32)
        t = np.linspace(0, 2, sr * 2, False)
        signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio = np.concatenate([silence, signal, silence])
        sf.write(SAMPLE_CLIP_PATH, audio, sr)


def test_strip_silence_from_filepath():
    """Test that strip_silence accepts an audio file path and returns a 1D float32 array."""
    assert os.path.exists(SAMPLE_CLIP_PATH)
    cleaned = strip_silence(SAMPLE_CLIP_PATH)
    assert isinstance(cleaned, np.ndarray)
    assert cleaned.ndim == 1
    assert cleaned.dtype == np.float32
    assert len(cleaned) > 0


def test_strip_silence_from_numpy_array():
    """Test that strip_silence accepts a direct 1D numpy waveform array."""
    sr = 16000
    dummy_waveform = np.sin(2 * np.pi * 220 * np.linspace(0, 1, sr)).astype(np.float32)
    cleaned = strip_silence(dummy_waveform)
    assert isinstance(cleaned, np.ndarray)
    assert cleaned.ndim == 1
    assert cleaned.dtype == np.float32


def test_strip_silence_stereo_conversion():
    """Test that multi-channel stereo input arrays are correctly converted to 1D mono."""
    sr = 16000
    stereo_waveform = np.random.randn(2, sr).astype(np.float32)
    cleaned = strip_silence(stereo_waveform)
    assert isinstance(cleaned, np.ndarray)
    assert cleaned.ndim == 1
    assert cleaned.dtype == np.float32
