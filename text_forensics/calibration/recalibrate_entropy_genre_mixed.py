"""
calibration/recalibrate_entropy_genre_mixed.py

Correction 6: Recalibrate Entropy with a Genre-Mixed Corpus.

Problem:
  HC3 human answers are short and conversational (mean entropy mu0=5.9088).
  Long-form technical writing naturally has high vocabulary richness (entropy > 7.0),
  causing the old entropy signal to misread technical writing (human or AI) as "0.33/100"
  (obviously human), overriding strong AI curvature signals.

Fix:
  Combines HC3 human answers (250 samples) with long-form technical/encyclopedic
  human articles from Wikitext-2 (200 samples >= 150 words).
  Computes genre-mixed mu0 and sigma0 specifically for entropy.
  Updates calibration/baseline_stats.json while leaving the other 3 signals untouched.
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.request
import numpy as np
from pathlib import Path

_CALIB_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _CALIB_DIR.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from text_forensics.signals.lexical_entropy import get_lexical_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_BASELINE_FILE = _CALIB_DIR / "baseline_stats.json"
_WIKITEXT_URL  = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"


def fetch_hc3_human_samples() -> list[str]:
    """Fetch HC3 human texts from HuggingFace cache or local test pool."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="Hello-SimpleAI/HC3", filename="all.jsonl", repo_type="dataset")
    
    human_texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            h_ans = item.get("human_answers", [])
            for ans in h_ans:
                words = ans.split()
                if 60 <= len(words) <= 500:
                    human_texts.append(ans)
                    if len(human_texts) >= 250:
                        break
            if len(human_texts) >= 250:
                break
    logger.info("Loaded %d HC3 human samples.", len(human_texts))
    return human_texts


def fetch_wikitext_technical_samples(target_count: int = 200) -> list[str]:
    """Fetch long-form technical/wikipedia human articles from Wikitext-2."""
    logger.info("Fetching Wikitext-2 technical articles from %s ...", _WIKITEXT_URL)
    req = urllib.request.Request(_WIKITEXT_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode("utf-8")

    lines = content.split("\n")
    blocks = []
    current = []
    curr_words = 0

    for l in lines:
        l_str = l.strip()
        if not l_str or (l_str.startswith("=") and l_str.endswith("=")):
            if curr_words >= 150:
                blocks.append(" ".join(current))
                if len(blocks) >= target_count:
                    break
            current = []
            curr_words = 0
        else:
            current.append(l_str)
            curr_words += len(l_str.split())
            if curr_words >= 250:
                blocks.append(" ".join(current))
                if len(blocks) >= target_count:
                    break
                current = []
                curr_words = 0

    logger.info("Extracted %d Wikitext-2 long-form technical blocks (>=150 words).", len(blocks))
    return blocks[:target_count]


def recalibrate_entropy():
    logger.info("=== Correction 6: Recalibrating Entropy Signal Baseline ===")

    # 1. Fetch sample pools
    hc3_samples = fetch_hc3_human_samples()
    wiki_samples = fetch_wikitext_technical_samples(200)
    all_samples = hc3_samples + wiki_samples
    logger.info("Combined sample pool: %d total human samples (%d HC3 + %d Wikitext).",
                len(all_samples), len(hc3_samples), len(wiki_samples))

    # 2. Compute entropy for all samples
    entropies = []
    for i, text in enumerate(all_samples):
        if (i + 1) % 100 == 0:
            logger.info("  Computing entropy [%d/%d] ...", i + 1, len(all_samples))
        stats = get_lexical_stats(text)
        entropies.append(stats["entropy"])

    entropies = np.array(entropies)
    new_mu0 = float(np.mean(entropies))
    new_sigma0 = float(np.std(entropies))

    logger.info("Calculated New Entropy Baseline: mu0=%.6f, sigma0=%.6f (n=%d)",
                new_mu0, new_sigma0, len(entropies))

    # 3. Load baseline_stats.json and update ONLY entropy
    with open(_BASELINE_FILE, encoding="utf-8") as f:
        baseline_stats = json.load(f)

    old_entropy = baseline_stats.get("entropy", {})
    logger.info("Old Entropy Baseline: mu0=%.6f, sigma0=%.6f (n=%s)",
                old_entropy.get("mu0", 0.0), old_entropy.get("sigma0", 1.0), old_entropy.get("n_valid"))

    baseline_stats["entropy"] = {
        "mu0": round(new_mu0, 6),
        "sigma0": round(new_sigma0, 6),
        "n_valid": len(entropies),
    }
    baseline_stats["entropy_calibration_version"] = "v2_genre_mixed"
    baseline_stats["entropy_source"] = f"HC3 ({len(hc3_samples)}) + Wikitext-2 ({len(wiki_samples)}) genre-mixed technical human text"

    with open(_BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(baseline_stats, f, indent=2)

    logger.info("Updated %s with v2_genre_mixed entropy baseline.", _BASELINE_FILE)


if __name__ == "__main__":
    recalibrate_entropy()
