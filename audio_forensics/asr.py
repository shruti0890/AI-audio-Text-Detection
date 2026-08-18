"""
Automatic Speech Recognition (ASR) Module
===========================================
Transcribes speech to text using Hugging Face transformers pipeline with Whisper-Tiny (openai/whisper-tiny).

Input contract: audio file path (str), 1D np.ndarray, or torch.Tensor (16kHz float32 waveform)
Output contract: plain string transcript with no metadata wrapping.
"""

import os
from typing import Union
import numpy as np
import torch
from transformers import pipeline

# Global module cache to prevent reloading model weights on every function invocation
_ASR_PIPELINE = None


def get_asr_pipeline():
    """Loads and caches the Hugging Face transformers ASR pipeline with openai/whisper-tiny."""
    global _ASR_PIPELINE
    if _ASR_PIPELINE is None:
        model_id = "openai/whisper-tiny"
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, local_files_only=True)
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            _ASR_PIPELINE = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,
                return_timestamps=False,
                generate_kwargs={"task": "transcribe", "language": "en"},
            )
        except Exception:
            _ASR_PIPELINE = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                chunk_length_s=30,
                return_timestamps=False,
                generate_kwargs={"task": "transcribe", "language": "en"},
            )
    return _ASR_PIPELINE


def transcribe(
    processed_audio: Union[str, np.ndarray, torch.Tensor],
    sampling_rate: int = 16000
) -> str:
    """Transcribes spoken audio into a plain text string using Whisper-Tiny.

    Args:
        processed_audio (Union[str, np.ndarray, torch.Tensor]): Audio file path or 1D audio array.
        sampling_rate (int): Sampling rate in Hz (default: 16000 Hz).

    Returns:
        str: Plain text transcription of spoken content with no metadata dictionary wrapping.
    """
    asr = get_asr_pipeline()

    # 1. Normalize input into format expected by Hugging Face ASR pipeline
    if isinstance(processed_audio, str):
        from .audio_loader import load_and_normalize_audio
        waveform, _ = load_and_normalize_audio(processed_audio, target_sr=sampling_rate)
        audio_input = {"raw": waveform.astype(np.float32), "sampling_rate": sampling_rate}
    elif isinstance(processed_audio, torch.Tensor):
        waveform = processed_audio.detach().cpu().numpy()
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
        audio_input = {"raw": np.nan_to_num(waveform, nan=0.0).astype(np.float32), "sampling_rate": sampling_rate}
    elif isinstance(processed_audio, np.ndarray):
        waveform = processed_audio
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
        audio_input = {"raw": np.nan_to_num(waveform, nan=0.0).astype(np.float32), "sampling_rate": sampling_rate}
    else:
        raise ValueError(f"Unsupported audio input type: {type(processed_audio)}")

    # 2. Run inference through Whisper-Tiny ASR pipeline
    result = asr(audio_input)

    # 3. Extract clean string transcript from pipeline dictionary output
    if isinstance(result, dict) and "text" in result:
        text = result["text"]
    elif isinstance(result, list) and len(result) > 0 and "text" in result[0]:
        text = result[0]["text"]
    else:
        text = str(result)

    return text.strip()
