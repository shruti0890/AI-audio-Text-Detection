import os
import numpy as np
import soundfile as sf
from audio_forensics.asr import transcribe

SAMPLE_CLIP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "sample_clips", "test_clip_with_silence.wav"
)
SPEECH_CLIP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "sample_clips", "speech_sample.wav"
)


def setup_module():
    """Ensure sample audio clips exist before running ASR tests."""
    os.makedirs(os.path.dirname(SAMPLE_CLIP_PATH), exist_ok=True)
    if not os.path.exists(SAMPLE_CLIP_PATH):
        sr = 16000
        silence = np.zeros(int(sr * 1.5), dtype=np.float32)
        t = np.linspace(0, 2, sr * 2, False)
        signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio = np.concatenate([silence, signal, silence])
        sf.write(SAMPLE_CLIP_PATH, audio, sr)


def test_transcribe_numpy_array():
    """Test that transcribe accepts a 16kHz float32 numpy array and returns a plain string."""
    sr = 16000
    dummy_waveform = np.sin(2 * np.pi * 220 * np.linspace(0, 1, sr)).astype(np.float32)
    transcript = transcribe(dummy_waveform)
    assert isinstance(transcript, str), f"Expected plain str, got {type(transcript)}"
    assert not isinstance(transcript, dict), "Transcript must not be wrapped in a dictionary"


def test_transcribe_filepath():
    """Test that transcribe accepts an audio filepath and returns a plain string."""
    assert os.path.exists(SAMPLE_CLIP_PATH)
    transcript = transcribe(SAMPLE_CLIP_PATH)
    assert isinstance(transcript, str), f"Expected plain str, got {type(transcript)}"
    assert not isinstance(transcript, dict), "Transcript must not be wrapped in a dictionary"


def test_transcribe_speech_clip():
    """Test that transcribe converts a spoken audio clip into a plain text transcript."""
    if os.path.exists(SPEECH_CLIP_PATH):
        transcript = transcribe(SPEECH_CLIP_PATH)
        assert isinstance(transcript, str), f"Expected plain str, got {type(transcript)}"
        assert not isinstance(transcript, dict), "Transcript must not be wrapped in a dictionary"
        assert len(transcript) > 0, "Expected non-empty transcript for spoken audio clip"

