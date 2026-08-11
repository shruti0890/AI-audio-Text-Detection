"""
audio_forensics/calibration/run_calibration.py
===============================================
Empirical threshold calibration for wav2vec2-deepfake-voice-detector.

Run from the repository root (Integration_branch):
    python audio_forensics/calibration/run_calibration.py

Expects clips in:
    audio_forensics/sample_clips/fake/   ← TTS-generated clips (ElevenLabs, Bark, XTTS v2, ...)
    audio_forensics/sample_clips/real/   ← Real human voice clips (your own, CommonVoice, VCTK, ...)

Outputs:
    audio_forensics/calibration/calibration_results.json  ← per-clip scores + ground truth
    audio_forensics/calibration/roc_data.json             ← full ROC curve + EER + FPR<1% threshold

After running, update model_config.json calibration_parameters.decision_threshold_pct
with the recommended threshold. No code changes needed — the pipeline reads the config
at startup.

See AUDIO_CALIBRATION_PROTOCOL.md for full instructions.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from audio_forensics.pipeline import analyze_audio

# ── Clip directories ──────────────────────────────────────────────────────────
_FAKE_DIR = _REPO_ROOT / "audio_forensics" / "sample_clips" / "fake"
_REAL_DIR = _REPO_ROOT / "audio_forensics" / "sample_clips" / "real"
_CALIB_DIR = _REPO_ROOT / "audio_forensics" / "calibration"

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_directory(directory: Path, label: int, label_name: str) -> list[dict]:
    """Score all audio files in a directory. label=1 for fake, label=0 for real."""
    results = []
    clips = sorted([f for f in directory.glob("*") if f.suffix.lower() in _AUDIO_EXTENSIONS])

    if not clips:
        print(f"  ⚠  No audio clips found in: {directory}")
        print(f"     Add .wav / .mp3 / .flac clips and re-run.")
        return results

    for clip in clips:
        try:
            out = analyze_audio(str(clip))
            score = out["audio_score"]
            threshold = out.get("decision_threshold_pct", 62.5)
            verdict = "FAKE" if score >= threshold else "REAL"
            correct = "✅" if (verdict == label_name) else "❌"
            print(f"  {correct} {label_name:4s}  {clip.name:<40}  score={score:6.2f}%  verdict={verdict}")
            results.append({
                "file": clip.name,
                "label": label,         # ground truth: 1=fake, 0=real
                "score": score,
                "threshold_used": threshold,
            })
        except Exception as exc:
            print(f"  ✗  ERROR on {clip.name}: {exc}")

    return results


def _compute_roc(results: list[dict]) -> tuple[list[dict], dict, dict | None]:
    """Sweep thresholds, compute ROC curve, return (roc_data, eer_row, fpr1_row)."""
    labels = np.array([r["label"] for r in results])
    scores = np.array([r["score"] for r in results])

    thresholds = np.arange(0.0, 100.5, 0.5)
    roc_data = []

    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))

        n_pos = tp + fn
        n_neg = fp + tn
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        fnr = fn / n_pos if n_pos > 0 else 0.0

        roc_data.append({
            "threshold": float(t),
            "tpr": round(tpr, 6),
            "fpr": round(fpr, 6),
            "fnr": round(fnr, 6),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })

    # EER: where |FPR - FNR| is minimised
    eer_row = min(roc_data, key=lambda r: abs(r["fpr"] - r["fnr"]))

    # FPR < 1%: most sensitive threshold where FPR < 0.01 and TPR > 0
    fpr1_candidates = [r for r in roc_data if r["fpr"] < 0.01 and r["tpr"] > 0]
    fpr1_row = max(fpr1_candidates, key=lambda r: r["tpr"]) if fpr1_candidates else None

    return roc_data, eer_row, fpr1_row


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Audio Deepfake Calibration Run")
    print("  Model: garystafford/wav2vec2-deepfake-voice-detector")
    print("=" * 70)

    all_results = []

    # Score fake clips
    print(f"\n[1/2] Scoring FAKE clips from: {_FAKE_DIR}")
    _FAKE_DIR.mkdir(parents=True, exist_ok=True)
    all_results += _score_directory(_FAKE_DIR, label=1, label_name="FAKE")

    # Score real clips
    print(f"\n[2/2] Scoring REAL clips from: {_REAL_DIR}")
    _REAL_DIR.mkdir(parents=True, exist_ok=True)
    all_results += _score_directory(_REAL_DIR, label=0, label_name="REAL")

    if len(all_results) < 4:
        print("\n⚠  Not enough clips to compute a reliable ROC curve.")
        print("   Add at least 2 fake clips and 2 real clips, then re-run.")
        return

    # Save raw results
    results_path = _CALIB_DIR / "calibration_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Raw scores saved: {results_path}")

    # Compute ROC
    roc_data, eer_row, fpr1_row = _compute_roc(all_results)

    n_fake = sum(1 for r in all_results if r["label"] == 1)
    n_real = sum(1 for r in all_results if r["label"] == 0)

    print(f"\n{'='*70}")
    print(f"  Dataset: {n_fake} fake clips | {n_real} real clips | {len(all_results)} total")
    print(f"{'='*70}")

    print(f"\n=== EQUAL ERROR RATE (EER) ===")
    print(f"  Threshold : {eer_row['threshold']:.1f}%")
    print(f"  EER       : {(eer_row['fpr'] + eer_row['fnr']) / 2:.4f}")
    print(f"  FPR       : {eer_row['fpr']:.4f}  ({eer_row['fp']}/{eer_row['fp']+eer_row['tn']} real clips wrongly flagged)")
    print(f"  FNR       : {eer_row['fnr']:.4f}  ({eer_row['fn']}/{eer_row['fn']+eer_row['tp']} fake clips missed)")

    if fpr1_row:
        print(f"\n=== FPR < 1% THRESHOLD (conservative, protects real speakers) ===")
        print(f"  Threshold : {fpr1_row['threshold']:.1f}%")
        print(f"  TPR       : {fpr1_row['tpr']:.4f}  ({fpr1_row['tp']}/{n_fake} fake clips caught)")
        print(f"  FPR       : {fpr1_row['fpr']:.4f}  ({fpr1_row['fp']}/{n_real} real clips wrongly flagged)")
        recommended = fpr1_row["threshold"]
    else:
        print(f"\n⚠  No threshold achieves FPR < 1% on this dataset.")
        print(f"   Using EER threshold as fallback.")
        recommended = eer_row["threshold"]

    print(f"\n{'='*70}")
    print(f"  ✅ RECOMMENDED decision_threshold_pct = {recommended}")
    print(f"\n  → Update audio_forensics/calibration/model_config.json:")
    print(f'    "decision_threshold_pct": {recommended},')
    print(f'    "calibration_status": "calibrated",')
    print(f"{'='*70}\n")

    # Save ROC data
    roc_path = _CALIB_DIR / "roc_data.json"
    roc_payload = {
        "n_fake": n_fake,
        "n_real": n_real,
        "eer": eer_row,
        "fpr1_threshold": recommended,
        "fpr1_row": fpr1_row,
        "roc_curve": roc_data,
    }
    with open(roc_path, "w") as f:
        json.dump(roc_payload, f, indent=2)
    print(f"✅ ROC data saved: {roc_path}")
    print(f"\n   Run AUDIO_CALIBRATION_PROTOCOL.md Step 4 to update model_config.json.")
    print(f"   Run plot_distribution.py to visualize score distributions.\n")


if __name__ == "__main__":
    main()
