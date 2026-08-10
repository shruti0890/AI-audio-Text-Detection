"""
Audio Forensics Package
======================
This package provides modular tools for voice activity detection (VAD),
speech-to-text transcription (ASR), and deepfake classification using pre-trained ML models.
"""

from .pipeline import analyze_audio

__all__ = ["analyze_audio"]
