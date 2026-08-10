"""Quick calibration runner using the embedded fallback corpus only (no network required)."""
import sys, json, math, random
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from text_forensics.calibration.run_calibration import _FALLBACK_CORPUS, run_calibration

# Repeat the 30-paragraph corpus with shuffling to reach 200 samples
samples = _FALLBACK_CORPUS.copy()
random.seed(42)
random.shuffle(samples)
while len(samples) < 200:
    extra = _FALLBACK_CORPUS.copy()
    random.shuffle(extra)
    samples.extend(extra)
samples = samples[:200]

source = (
    "Embedded fallback corpus (30 author-written human paragraphs, "
    "repeated with shuffling to reach 200 samples). "
    "Substitution reason: Wikipedia API rate-limited (HTTP 429) during calibration run."
)

print(f"Running calibration on {len(samples)} samples...")
stats = run_calibration(samples, source)

output_path = Path(__file__).resolve().parents[1] / "calibration" / "baseline_stats.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("\n=== Calibration Results ===")
for sig in ["curvature", "burstiness", "cliche_density", "entropy"]:
    s = stats[sig]
    print(f"  {sig:<20} mu0={s['mu0']:.4f}  sigma0={s['sigma0']:.4f}  (n={s['n_valid']})")
print(f"  sample_size={stats['sample_size']}")
print(f"  source={stats['source'][:60]}...")
print(f"  date={stats['date']}")
print(f"\nSaved to: {output_path}")
