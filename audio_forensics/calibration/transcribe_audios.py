"""
audio_forensics/calibration/transcribe_audios.py
================================================
Transcribes audio files from Cat1, Cat2, Cat3, and Cat4 directories using
the actual Whisper-Tiny ASR pipeline (preprocessed with Silero VAD silence stripping,
matching the live audio forensics pipeline flow), and saves/updates the transcribed
text into calibration_results.json corresponding to each file name.

Run from repository root:
    python audio_forensics/calibration/transcribe_audios.py
"""

import json
import os
import sys
from pathlib import Path

# Set repository root in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio_forensics.vad import strip_silence
from audio_forensics.asr import transcribe

# Calibration file path
_CALIB_RESULTS_PATH = _REPO_ROOT / "audio_forensics" / "calibration" / "calibration_results.json"

# Category directories to scan
_CATEGORY_DIRS = [
    _REPO_ROOT / "Cat1_human voice_human txt",
    _REPO_ROOT / "Cat2_ai voice_ai text",
    _REPO_ROOT / "Cat3_human voice_ai txt",
    _REPO_ROOT / "Cat4_ai voice_human txt",
]

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def transcribe_all_categories():
    print("=" * 70)
    print("  ASR Transcription for Cat1, Cat2, Cat3, Cat4 Audio Clips")
    print("  Model: OpenAI Whisper-Tiny (with Silero VAD preprocessing)")
    print("=" * 70)

    # 1. Load existing calibration results if available
    existing_results = []
    if _CALIB_RESULTS_PATH.exists():
        try:
            with open(_CALIB_RESULTS_PATH, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            print(f"Loaded {len(existing_results)} existing records from {_CALIB_RESULTS_PATH.name}")
        except Exception as e:
            print(f"Warning loading {_CALIB_RESULTS_PATH.name}: {e}. Starting fresh list.")
            existing_results = []

    # Map file -> record
    results_map = {}
    for item in existing_results:
        if isinstance(item, dict) and "file" in item:
            results_map[item["file"]] = item

    total_transcribed = 0

    # 2. Iterate through all category directories
    for cat_dir in _CATEGORY_DIRS:
        if not cat_dir.exists():
            print(f"\n⚠  Directory not found: {cat_dir.name}")
            continue

        audio_files = sorted([f for f in cat_dir.iterdir() if f.suffix.lower() in _AUDIO_EXTENSIONS])
        print(f"\n[*] Processing {cat_dir.name} ({len(audio_files)} audio files)...")

        for idx, file_path in enumerate(audio_files, 1):
            file_name = file_path.name
            try:
                # Mirroring actual audio pipeline: strip silence with VAD -> transcribe with Whisper-Tiny
                processed_waveform = strip_silence(str(file_path))
                text_transcript = transcribe(processed_waveform)

                if file_name in results_map:
                    results_map[file_name]["transcript"] = text_transcript
                else:
                    # If not present in existing json, create entry
                    # Guess default label: Cat1/Cat3 = Human voice (0), Cat2/Cat4 = AI voice (1)
                    is_fake = 1 if ("cat2" in file_name.lower() or "cat4" in file_name.lower()) else 0
                    results_map[file_name] = {
                        "file": file_name,
                        "label": is_fake,
                        "score": None,
                        "threshold_used": 62.5,
                        "transcript": text_transcript,
                    }

                total_transcribed += 1
                snippet = text_transcript[:60] + "..." if len(text_transcript) > 60 else text_transcript
                print(f"  [{idx}/{len(audio_files)}] {file_name:<25} -> \"{snippet}\"")

            except Exception as exc:
                print(f"  [!] [{idx}/{len(audio_files)}] ERROR transcribing {file_name}: {exc}")

    # 3. Save back into calibration_results.json preserving order or updated structure
    # If existing list had items, preserve original order and append any new ones
    final_results = []
    seen_files = set()

    for item in existing_results:
        f_name = item.get("file")
        if f_name in results_map:
            final_results.append(results_map[f_name])
            seen_files.add(f_name)
        else:
            final_results.append(item)

    for f_name, item in results_map.items():
        if f_name not in seen_files:
            final_results.append(item)

    with open(_CALIB_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Successfully transcribed {total_transcribed} files.")
    print(f"[SUCCESS] Saved updated results to {_CALIB_RESULTS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    transcribe_all_categories()

