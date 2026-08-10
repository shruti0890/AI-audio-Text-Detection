"""
Voice Activity Detection (VAD) Module
======================================
Strips silence from input audio files or live recordings using Silero VAD.

Input contract: audio file path (str), np.ndarray, or torch.Tensor
Output contract: 1D np.ndarray containing 16kHz float32 mono PCM waveform samples stripped of silence.
"""

import os
from typing import Union
import numpy as np
import torch
import librosa

# Global module cache to prevent reloading the model on every function invocation
_VAD_MODEL = None
_VAD_UTILS = None


def get_vad_model():
    """Loads and caches the Silero VAD model and helper utilities via torch.hub."""
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None or _VAD_UTILS is None:
        _VAD_MODEL, _VAD_UTILS = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            skip_validation=True,
            onnx=False
        )
    return _VAD_MODEL, _VAD_UTILS


def strip_silence(
    audio_input: Union[str, np.ndarray, torch.Tensor],
    sampling_rate: int = 16000,
    threshold: float = 0.5
) -> np.ndarray:
    """Strips non-speech and silent portions from audio using Silero VAD.

    Args:
        audio_input (Union[str, np.ndarray, torch.Tensor]): Path to audio file or raw audio waveform.
        sampling_rate (int): Target sampling rate in Hz (default: 16000 Hz).
        threshold (float): Speech confidence probability threshold for Silero VAD (0.0 to 1.0).

    Returns:
        np.ndarray: 1D float32 numpy array sampled at 16kHz with silence removed.
    """
    model, utils = get_vad_model()
    get_speech_timestamps, _, _, _, collect_chunks = utils

    # 1. Load audio and normalize to 16kHz mono float32 waveform
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        # librosa automatically handles format decoding (.mp3, .wav), mono conversion, and 16kHz resampling
        waveform, _ = librosa.load(audio_input, sr=sampling_rate, mono=True)
    elif isinstance(audio_input, torch.Tensor):
        waveform = audio_input.detach().cpu().numpy()
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)  # Convert multi-channel/stereo to mono
    elif isinstance(audio_input, np.ndarray):
        waveform = audio_input
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)  # Convert multi-channel/stereo to mono
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    waveform = waveform.astype(np.float32)

    # Handle empty input edge case
    if len(waveform) == 0:
        return np.zeros(sampling_rate, dtype=np.float32)

    # Convert waveform to PyTorch Tensor expected by Silero VAD
    wav_tensor = torch.from_numpy(waveform)

    # 2. Extract speech start and end timestamps from Silero VAD
    speech_timestamps = get_speech_timestamps(
        wav_tensor,
        model,
        sampling_rate=sampling_rate,
        threshold=threshold
    )

    # 3. Trim non-speech segments and concatenate speech chunks
    if speech_timestamps:
        cleaned_tensor = collect_chunks(speech_timestamps, wav_tensor)
        cleaned_waveform = cleaned_tensor.numpy()
    else:
        # Fallback if no active speech is detected (e.g., pure silence clip)
        cleaned_waveform = waveform

    return cleaned_waveform.astype(np.float32)
