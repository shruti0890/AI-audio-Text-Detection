"""
audio_forensics/audio_loader.py
===============================
Unified Audio Normalization and Format Decoding Module.

Supports:
    .wav, .mp3, .flac, .ogg, .m4a, .aac

Provides a single-pass decoding pipeline that converts any supported audio file
or waveform into a standard internal representation:
    - 16,000 Hz Sampling Rate
    - 1D Mono Channel (averaged if stereo/multi-channel)
    - float32 NumPy array (values normalized to [-1.0, 1.0])
    - Full extracted metadata dictionary

Handles external decoder (FFmpeg) fallback and produces clean, informative error
messages instead of raw NoBackendError exceptions.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Union

import numpy as np
import soundfile as sf
import librosa

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


class AudioDecodingError(RuntimeError):
    """Raised when an audio file cannot be decoded or is corrupted."""
    pass


class UnsupportedAudioFormatError(ValueError):
    """Raised when an audio format is not supported."""
    pass


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg executable is available on the system PATH."""
    return shutil.which("ffmpeg") is not None


def _decode_with_ffmpeg(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """Decodes audio using ffmpeg subprocess directly to 16kHz mono float32 raw PCM."""
    if not is_ffmpeg_available():
        ext = os.path.splitext(file_path)[1].lower()
        raise AudioDecodingError(
            f"Audio decoding is unavailable for format '{ext}'. "
            f"FFmpeg is required to decode {ext.upper()} files on this system but was not found in PATH. "
            f"Please install FFmpeg or upload audio in WAV, MP3, FLAC, or OGG format."
        )

    # Convert to temporary wav using ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", file_path,
            "-vn",
            "-ac", "1",
            "-ar", str(target_sr),
            "-f", "wav",
            tmp_wav_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise AudioDecodingError(
                f"FFmpeg failed to decode '{os.path.basename(file_path)}': {res.stderr.strip()}"
            )

        waveform, _ = sf.read(tmp_wav_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return waveform.astype(np.float32)
    except Exception as exc:
        if isinstance(exc, AudioDecodingError):
            raise
        raise AudioDecodingError(f"Error during FFmpeg decoding of {file_path}: {exc}") from exc
    finally:
        if os.path.exists(tmp_wav_path):
            try:
                os.remove(tmp_wav_path)
            except OSError:
                pass


def load_and_normalize_audio(
    audio_input: Union[str, Path, np.ndarray],
    target_sr: int = 16000,
    orig_filename: str = ""
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads, decodes, and normalizes audio into 16kHz mono float32 PCM.

    Args:
        audio_input: File path (str/Path) or existing numpy waveform array.
        target_sr: Target sampling rate in Hz (default: 16000).
        orig_filename: Optional original filename if passing array or temp file.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - 1D float32 numpy waveform array sampled at `target_sr`
            - Metadata dictionary containing:
                * original_filename: str
                * original_format: str (e.g. 'WAV', 'MP3', 'M4A', etc.)
                * original_duration_seconds: float
                * original_sample_rate: int
                * original_channels: int
                * normalized_sample_rate: int
                * normalized_channels: int
                * normalized_duration_seconds: float
                * total_samples: int
    """
    # ── 1. Handle in-memory numpy array input ─────────────────────────────────
    if isinstance(audio_input, np.ndarray):
        waveform = audio_input.astype(np.float32)
        orig_channels = 1
        if waveform.ndim > 1:
            orig_channels = waveform.shape[0] if waveform.shape[0] < waveform.shape[1] else waveform.shape[1]
            waveform = waveform.mean(axis=0 if waveform.shape[0] < waveform.shape[1] else 1)
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        duration = float(len(waveform) / target_sr) if target_sr > 0 else 0.0

        metadata = {
            "original_filename": orig_filename or "in_memory_audio",
            "original_format": "RAW_NUMPY",
            "original_duration_seconds": round(duration, 3),
            "original_sample_rate": target_sr,
            "original_channels": orig_channels,
            "normalized_sample_rate": target_sr,
            "normalized_channels": 1,
            "normalized_duration_seconds": round(duration, 3),
            "total_samples": len(waveform),
        }
        return waveform, metadata

    # ── 2. Handle File Path Input ─────────────────────────────────────────────
    file_path = str(audio_input)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    filename = orig_filename or os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
        )

    # ── Try reading initial metadata / decoding with soundfile / librosa ─────
    waveform = None
    orig_sr = target_sr
    orig_channels = 1
    orig_duration = 0.0

    try:
        # First attempt: soundfile inspect
        info = sf.info(file_path)
        orig_sr = info.samplerate
        orig_channels = info.channels
        orig_duration = info.duration
    except Exception:
        pass

    # Attempt 1: Standard librosa / soundfile decoding
    try:
        raw_waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)
        waveform = raw_waveform.astype(np.float32)
        if orig_duration == 0.0:
            orig_duration = len(waveform) / target_sr
    except Exception as librosa_exc:
        # Attempt 2: If M4A / AAC or librosa fails due to backend, fallback to FFmpeg
        if ext in {".m4a", ".aac"} or "NoBackendError" in str(type(librosa_exc)) or "audioread" in str(librosa_exc).lower():
            try:
                waveform = _decode_with_ffmpeg(file_path, target_sr=target_sr)
                if orig_duration == 0.0:
                    orig_duration = len(waveform) / target_sr
            except Exception as ffmpeg_exc:
                raise AudioDecodingError(
                    f"Unable to decode '{filename}'. The file may be corrupted or missing required decoders: {ffmpeg_exc}"
                ) from ffmpeg_exc
        else:
            raise AudioDecodingError(
                f"Failed to decode audio file '{filename}': {librosa_exc}"
            ) from librosa_exc

    if waveform is None or len(waveform) == 0:
        raise AudioDecodingError(f"Decoded audio file '{filename}' contains no audio samples.")

    # Clean non-finite numbers and ensure 1D float32
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    normalized_duration = float(len(waveform) / target_sr)

    metadata = {
        "original_filename": filename,
        "original_format": ext.replace(".", "").upper(),
        "original_duration_seconds": round(orig_duration or normalized_duration, 2),
        "original_sample_rate": orig_sr,
        "original_channels": orig_channels,
        "normalized_sample_rate": target_sr,
        "normalized_channels": 1,
        "normalized_duration_seconds": round(normalized_duration, 2),
        "total_samples": len(waveform),
    }

    return waveform, metadata
