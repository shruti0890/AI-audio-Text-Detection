"""
calibration/run_calibration_hc3.py

Correction 1: Re-calibrate using Hello-SimpleAI/HC3 human answers.

Dataset: Hello-SimpleAI/HC3 (all.jsonl) — downloaded via hf_hub_download.
  HC3 stores data as JSONL files (no parquet, datasets 5.x script-mode disabled).
  We use hf_hub_download() to fetch all.jsonl directly and parse it ourselves.

Fallback: Salesforce/wikitext wikitext-103-raw-v1 via datasets library.
  Human-only; no AI pairs available from this source.

The script:
  - Loads human_answers and chatgpt_answers from HC3.
  - Filters to texts with 60–500 words.
  - Shuffles, then splits:
      first N_CALIBRATION_SAMPLES → calibration (mu0/sigma0)
      next N_TEST_POOL            → held-out human test pool (Phase 1.8)
      chatgpt first N_TEST_POOL   → held-out AI test pool (Phase 1.8)
  - Runs all 4 signals on calibration samples.
  - Writes calibration/baseline_stats.json.
  - Writes calibration/hc3_test_pool_human.json
  - Writes calibration/hc3_test_pool_ai.json

Usage:
    python text_forensics/calibration/run_calibration_hc3.py
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
from datetime import date
from pathlib import Path
from typing import Optional

_PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ_ROOT))

from text_forensics.signals.burstiness import get_burstiness
from text_forensics.signals.cliche_scanner import get_cliche_density
from text_forensics.signals.curvature import get_curvature
from text_forensics.signals.lexical_entropy import get_lexical_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_CALIBRATION_SAMPLES = 250
N_TEST_POOL = 60
MIN_WORDS = 60
MAX_WORDS = 500

_CALIB_OUT  = Path(__file__).parent / "baseline_stats.json"
_HUMAN_POOL = Path(__file__).parent / "hc3_test_pool_human.json"
_AI_POOL    = Path(__file__).parent / "hc3_test_pool_ai.json"


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_hc3_direct() -> tuple[list[str], list[str], str]:
    """
    Download HC3 all.jsonl via hf_hub_download and parse human/AI answers.

    HC3 has 36k+ human answers and 26k+ AI (ChatGPT) answers.
    Filters to 60–500 words. Truncates to MAX_WORDS if needed.

    Returns:
        (human_texts, ai_texts, source_label)
    """
    from huggingface_hub import hf_hub_download

    logger.info("Downloading Hello-SimpleAI/HC3 all.jsonl via hf_hub_download...")
    jsonl_path = hf_hub_download(
        repo_id="Hello-SimpleAI/HC3",
        filename="all.jsonl",
        repo_type="dataset",
    )
    logger.info("Downloaded to: %s", jsonl_path)

    human_texts: list[str] = []
    ai_texts: list[str] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            for ans in (row.get("human_answers") or []):
                if isinstance(ans, str) and ans.strip():
                    words = ans.split()
                    if len(words) >= MIN_WORDS:
                        human_texts.append(" ".join(words[:MAX_WORDS]))

            for ans in (row.get("chatgpt_answers") or []):
                if isinstance(ans, str) and ans.strip():
                    words = ans.split()
                    if len(words) >= MIN_WORDS:
                        ai_texts.append(" ".join(words[:MAX_WORDS]))

    logger.info(
        "HC3 loaded: %d human texts, %d AI texts (after %d–%d word filter)",
        len(human_texts), len(ai_texts), MIN_WORDS, MAX_WORDS
    )
    return human_texts, ai_texts, "Hello-SimpleAI/HC3 all.jsonl (human_answers field)"


def _load_wikitext103_fallback() -> tuple[list[str], list[str], str]:
    """
    Fallback: Salesforce/wikitext wikitext-103-raw-v1 via datasets library.
    Returns human-only texts (no AI pairs from this source).
    """
    # pyrefly: ignore [missing-import]
    from datasets import load_dataset
    logger.info("Loading Salesforce/wikitext wikitext-103-raw-v1 as fallback...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")

    human_texts: list[str] = []
    current_words: list[str] = []

    for row in ds:
        line = (row.get("text") or "").strip()
        if not line:
            continue
        words = line.split()
        current_words.extend(words)
        while len(current_words) >= MIN_WORDS:
            chunk = current_words[:MAX_WORDS]
            human_texts.append(" ".join(chunk))
            current_words = current_words[MAX_WORDS:]
        if len(human_texts) >= N_CALIBRATION_SAMPLES + N_TEST_POOL + 100:
            break

    logger.info("wikitext-103 loaded: %d human texts (no AI pairs)", len(human_texts))
    return human_texts, [], "Salesforce/wikitext wikitext-103-raw-v1 (fallback; no AI pairs)"


def load_datasets() -> tuple[list[str], list[str], str]:
    """Load HC3 directly, fall back to wikitext-103 on error."""
    try:
        return _load_hc3_direct()
    except Exception as e:
        logger.warning("HC3 direct load failed: %s. Trying wikitext-103 fallback...", e)
        try:
            return _load_wikitext103_fallback()
        except Exception as e2:
            raise RuntimeError(
                f"Both HC3 and wikitext-103 failed.\nHC3: {e}\nWikitext: {e2}"
            )


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def _std(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 1.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / n)


# ---------------------------------------------------------------------------
# Signal runner
# ---------------------------------------------------------------------------

def run_signals_on_samples(samples: list[str]) -> dict[str, list[float]]:
    """Run all 4 signals on samples and return raw value lists."""
    curvature_vals: list[float] = []
    burstiness_vals: list[float] = []
    cliche_vals: list[float] = []
    entropy_vals: list[float] = []

    n = len(samples)
    for i, text in enumerate(samples):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info("  [%d/%d] processing...", i + 1, n)

        try:
            c = get_curvature(text)
            if c is not None:
                curvature_vals.append(c)
        except Exception as e:
            logger.warning("Curvature error sample %d: %s", i + 1, e)

        try:
            b = get_burstiness(text)
            if b is not None:
                burstiness_vals.append(b)
        except Exception as e:
            logger.warning("Burstiness error sample %d: %s", i + 1, e)

        try:
            cliche_vals.append(get_cliche_density(text))
        except Exception as e:
            logger.warning("Cliche error sample %d: %s", i + 1, e)

        try:
            lex = get_lexical_stats(text)
            entropy_vals.append(lex["entropy"])
        except Exception as e:
            logger.warning("Entropy error sample %d: %s", i + 1, e)

    return {
        "curvature":      curvature_vals,
        "burstiness":     burstiness_vals,
        "cliche_density": cliche_vals,
        "entropy":        entropy_vals,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)

    all_human, all_ai, source_label = load_datasets()

    # Shuffle independently
    random.shuffle(all_human)
    if all_ai:
        random.shuffle(all_ai)

    logger.info("Available: %d human, %d AI", len(all_human), len(all_ai))

    needed = N_CALIBRATION_SAMPLES + N_TEST_POOL
    if len(all_human) < needed:
        logger.warning("Only %d human samples; needed %d. Using all.", len(all_human), needed)

    # Split: calibration | test pool (strictly non-overlapping)
    calib_samples   = all_human[:N_CALIBRATION_SAMPLES]
    test_human_pool = all_human[N_CALIBRATION_SAMPLES : N_CALIBRATION_SAMPLES + N_TEST_POOL]
    test_ai_pool    = all_ai[:N_TEST_POOL] if all_ai else []

    logger.info(
        "Split → calibration: %d | test human: %d | test AI: %d",
        len(calib_samples), len(test_human_pool), len(test_ai_pool)
    )

    # Run signals
    logger.info("Running 4 signals on %d calibration samples...", len(calib_samples))
    raw = run_signals_on_samples(calib_samples)

    # Build baseline_stats.json
    stats = {
        sig: {
            "mu0":     round(_mean(raw[sig]), 6),
            "sigma0":  round(max(_std(raw[sig]), 1e-6), 6),
            "n_valid": len(raw[sig]),
        }
        for sig in ["curvature", "burstiness", "cliche_density", "entropy"]
    }
    stats["sample_size"] = len(calib_samples)
    stats["source"]      = source_label
    stats["date"]        = date.today().isoformat()

    _CALIB_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_CALIB_OUT, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved baseline_stats.json")

    with open(_HUMAN_POOL, "w", encoding="utf-8") as f:
        json.dump(test_human_pool, f, indent=2, ensure_ascii=False)
    with open(_AI_POOL, "w", encoding="utf-8") as f:
        json.dump(test_ai_pool, f, indent=2, ensure_ascii=False)
    logger.info("Saved hc3_test_pool_human.json (%d) and hc3_test_pool_ai.json (%d)",
                len(test_human_pool), len(test_ai_pool))

    logger.info("\n=== Calibration Results ===")
    for sig in ["curvature", "burstiness", "cliche_density", "entropy"]:
        s = stats[sig]
        logger.info("  %-20s mu0=%8.4f  sigma0=%8.4f  (n=%d)",
                    sig, s["mu0"], s["sigma0"], s["n_valid"])
    logger.info("  sample_size=%d  source=%s", stats["sample_size"], stats["source"][:70])
