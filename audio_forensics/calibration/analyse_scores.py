import json
from pathlib import Path
from statistics import mean, median, multimode
import numpy as np

# Path to the calibration results file
RESULTS_PATH = Path("audio_forensics/calibration/calibration_results.json")

if not RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"Calibration results not found: {RESULTS_PATH}\\n"
        "Run run_calibration.py first."
    )

# Load results
with open(RESULTS_PATH, "r") as f:
    results = json.load(f)

# Separate real and fake scores
real_scores = [r["score"] for r in results if r["label"] == 0]
fake_scores = [r["score"] for r in results if r["label"] == 1]


def print_stats(name, scores):
    if not scores:
        print(f"No scores found for {name}.")
        return

    modes = multimode(scores)

    print(f"\\n=== {name} ===")
    print(f"Count              : {len(scores)}")
    print(f"Mean               : {mean(scores):.2f}")
    print(f"Median             : {median(scores):.2f}")
    print(f"Mode(s)            : {[round(m,2) for m in modes]}")
    print(f"Minimum            : {min(scores):.2f}")
    print(f"Maximum            : {max(scores):.2f}")
    print(f"Standard deviation : {np.std(scores):.2f}")

    # Helpful percentiles
    print(f"25th percentile    : {np.percentile(scores,25):.2f}")
    print(f"75th percentile    : {np.percentile(scores,75):.2f}")


print_stats("REAL (human) clips", real_scores)
print_stats("FAKE (AI) clips", fake_scores)

# Measure overlap
if real_scores and fake_scores:
    overlap = sum(1 for r in real_scores if r >= min(fake_scores))
    print("\\n=== DISTRIBUTION OVERLAP ===")
    print(f"Real clips scoring above the lowest fake score: {overlap}/{len(real_scores)}")

    real_mean = mean(real_scores)
    fake_mean = mean(fake_scores)
    print(f"Difference between means: {fake_mean - real_mean:.2f} points")

    print("\\nInterpretation:")
    if fake_mean - real_mean > 30:
        print("  Good separation between real and fake scores.")
    elif fake_mean - real_mean > 15:
        print("  Moderate separation; calibration may still work.")
    else:
        print("  Poor separation; model may be confusing real and fake audio on this dataset.")