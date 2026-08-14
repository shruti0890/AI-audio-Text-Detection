"""
Correction 15 — Phase F: Validation Against Known Failure Cases.

Tests:
  1. Bar Council / Manan Kumar Mishra news article (Correction 14 failure — scored 78.58, "AI-Generated")
  2. Transformers Wikipedia technical article (Correction 6 failure — entropy genre mismatch)
  3. 15-sample conversational HC3 regression check (ensure conversational still works)

Reports before/after verdicts for the two known failures.

Run from project root:
    python text_forensics/calibration/correction15_phase_f.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_CALIB_DIR = Path(__file__).resolve().parent
_PROJ_ROOT  = _CALIB_DIR.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from text_forensics.pipeline import _load_baseline_stats
from text_forensics.signals.curvature       import get_curvature
from text_forensics.signals.burstiness      import get_burstiness
from text_forensics.signals.cliche_scanner  import get_cliche_density
from text_forensics.signals.lexical_entropy import get_lexical_stats
from text_forensics.fusion                  import _cdf_score

# Known failure cases
BAR_COUNCIL_TEXT = (
    "Manan Kumar Mishra, the Chairperson of the Bar Council of India, "
    "has raised serious concerns about the increasing influence of artificial "
    "intelligence in the legal profession. Speaking at a national legal "
    "conference in New Delhi, he emphasised that while technology can assist "
    "lawyers, it must never replace human judgment, ethical reasoning, and "
    "the fundamental duty of an advocate to their client. "
    "Mishra warned that unregulated AI tools in courtrooms could undermine "
    "the adversarial system and compromise the principles of natural justice. "
    "He called on the Bar Council to frame guidelines for the responsible use "
    "of AI in legal practice, ensuring that accountability remains with the "
    "licensed advocate, not the algorithm."
)

TRANSFORMERS_TEXT = (
    "Transformer models have become the dominant architecture in natural language "
    "processing since the publication of the seminal paper 'Attention Is All You Need' "
    "by Vaswani et al. in 2017. The core innovation was the self-attention mechanism, "
    "which allows the model to weigh the relevance of each token in the input sequence "
    "when encoding each token's representation. Unlike recurrent neural networks, which "
    "process tokens sequentially, transformers process all tokens in parallel, enabling "
    "efficient training on modern GPU hardware. The encoder-decoder architecture was "
    "originally designed for sequence-to-sequence tasks such as machine translation, "
    "but subsequent work showed that encoder-only models (BERT) and decoder-only models "
    "(GPT) could achieve state-of-the-art results on a wide range of language understanding "
    "and generation tasks respectively. Pre-training on large unlabeled corpora followed by "
    "fine-tuning on downstream tasks became the dominant paradigm, dramatically reducing "
    "the amount of labeled data required for competitive performance across benchmarks."
)

# Before-state verdicts (Correction 14 weights, for comparison)
C14_WEIGHTS   = {"curvature": 0.5389, "burstiness": 0.1617, "cliche": 0.2694, "entropy": 0.03}
C14_T_HUMAN   = 66.131
C14_T_AI      = 73.838


def _calibrate(k: str, raw_val: float | None, baseline: dict) -> float | None:
    """Apply Gaussian CDF calibration to a raw signal value."""
    if raw_val is None:
        return None
    _INVERTED = {"burstiness", "entropy"}
    b_key = "cliche_density" if k == "cliche" else k
    stats  = baseline.get(b_key, {})
    mu0    = stats.get("mu0",    0.0)
    sig0   = stats.get("sigma0", 1.0)
    cdf    = _cdf_score(raw_val, mu0, sig0)
    v      = (100.0 - cdf) if k in _INVERTED else cdf
    return round(max(0.0, min(100.0, v)), 2)


def _score_text(text: str, weights: dict, baseline: dict) -> tuple[float, dict]:
    """Score text with given weights and baseline, return (fused_score, sub_scores)."""
    curv_raw    = get_curvature(text)
    burst_raw   = get_burstiness(text)
    cliche_raw  = get_cliche_density(text)
    ent_raw     = get_lexical_stats(text)["entropy"]

    subs = {
        "curvature":  _calibrate("curvature",  curv_raw,   baseline),
        "burstiness": _calibrate("burstiness", burst_raw,  baseline),
        "cliche":     _calibrate("cliche",     cliche_raw, baseline),
        "entropy":    _calibrate("entropy",    ent_raw,    baseline),
    }

    avail_w = {k: weights[k] for k in weights if subs[k] is not None}
    tot_w   = sum(avail_w.values())
    fused   = sum((avail_w[k] / tot_w) * subs[k] for k in avail_w) if tot_w > 0 else 50.0
    return round(fused, 2), subs


def _verdict(score: float, t_human: float, t_ai: float) -> str:
    if score >= t_ai:
        return f"Likely AI-Generated (>= {t_ai:.2f})"
    elif score >= t_human:
        return f"Uncertain / Mixed ({t_human:.2f} – {t_ai:.2f})"
    else:
        return f"Likely Human-Written (< {t_human:.2f})"


def run_phase_f() -> None:
    logger.info("=== Phase F: Validation Against Known Failure Cases ===")

    # Load updated config (C15 weights + thresholds)
    config_path = _CALIB_DIR / "fusion_config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    c15_weights = config["weights"]
    c15_t_human = config["thresholds"]["human_max"]
    c15_t_ai    = config["thresholds"]["ai_min"]

    # Load updated baseline stats (genre-diverse)
    baseline = _load_baseline_stats()

    logger.info("C15 weights loaded: %s", c15_weights)
    logger.info("C15 thresholds: t_human=%.4f, t_ai=%.4f", c15_t_human, c15_t_ai)

    # -------------------------------------------------------------------
    # Test 1: Bar Council news article
    # -------------------------------------------------------------------
    logger.info("\n--- Test 1: Bar Council News Article ---")
    logger.info("(%d words)", len(BAR_COUNCIL_TEXT.split()))

    c15_score, c15_subs = _score_text(BAR_COUNCIL_TEXT, c15_weights, baseline)
    c15_verdict = _verdict(c15_score, c15_t_human, c15_t_ai)

    logger.info("C14 before: score=78.58 | VERDICT: Likely AI-Generated (curvature dominated at 95.08)")
    logger.info("C15 after : score=%.2f | VERDICT: %s", c15_score, c15_verdict)
    logger.info("C15 sub-scores: curvature=%.1f  burstiness=%s  cliche=%.1f  entropy=%.1f",
                c15_subs["curvature"] or -1,
                f"{c15_subs['burstiness']:.1f}" if c15_subs["burstiness"] is not None else "None (< 5 sent)",
                c15_subs["cliche"] or -1,
                c15_subs["entropy"] or -1)

    bar_council_pass = c15_score < c15_t_ai  # At least not AI-Generated
    logger.info("✅ PASS (not AI-Generated)" if bar_council_pass else "❌ FAIL (still AI-Generated)")

    # -------------------------------------------------------------------
    # Test 2: Transformers Wikipedia article
    # -------------------------------------------------------------------
    logger.info("\n--- Test 2: Transformers / Wikipedia Technical Article ---")
    logger.info("(%d words)", len(TRANSFORMERS_TEXT.split()))

    c15_score2, c15_subs2 = _score_text(TRANSFORMERS_TEXT, c15_weights, baseline)
    c15_verdict2 = _verdict(c15_score2, c15_t_human, c15_t_ai)

    logger.info("C6  before: entropy signal caused genre mismatch (technical → high AI score)")
    logger.info("C14 before: entropy capped to 0.03 but curvature still dominated")
    logger.info("C15 after : score=%.2f | VERDICT: %s", c15_score2, c15_verdict2)
    logger.info("C15 sub-scores: curvature=%.1f  burstiness=%s  cliche=%.1f  entropy=%.1f",
                c15_subs2["curvature"] or -1,
                f"{c15_subs2['burstiness']:.1f}" if c15_subs2["burstiness"] is not None else "None (< 5 sent)",
                c15_subs2["cliche"] or -1,
                c15_subs2["entropy"] or -1)

    transformer_pass = c15_score2 < c15_t_ai
    logger.info("✅ PASS (not AI-Generated)" if transformer_pass else "❌ FAIL (still AI-Generated)")

    # -------------------------------------------------------------------
    # Test 3: HC3 conversational regression (15 samples)
    # -------------------------------------------------------------------
    logger.info("\n--- Test 3: Conversational Regression Check (15 HC3 samples) ---")

    with open(_CALIB_DIR / "hc3_test_pool_human.json", encoding="utf-8") as f:
        hc3_human = json.load(f)
    with open(_CALIB_DIR / "hc3_test_pool_ai.json", encoding="utf-8") as f:
        hc3_ai = json.load(f)

    # Use middle 15 samples from each (avoid the holdout tail used in Phase D)
    reg_human = hc3_human[25:40]  # 15 human samples
    reg_ai    = hc3_ai[25:40]     # 15 AI samples
    reg_texts = reg_human + reg_ai
    reg_labels = ["HUMAN"] * 15 + ["AI"] * 15

    correct = 0
    human_correct = 0
    ai_correct    = 0

    for i, (text, true_label) in enumerate(zip(reg_texts, reg_labels)):
        score, _ = _score_text(text, c15_weights, baseline)
        verdict  = _verdict(score, c15_t_human, c15_t_ai)
        predicted = "AI" if score >= c15_t_ai else ("HUMAN" if score < c15_t_human else "MIXED")
        is_correct = (
            (true_label == "AI"    and predicted == "AI")   or
            (true_label == "HUMAN" and predicted == "HUMAN")
        )
        if is_correct:
            correct += 1
        if true_label == "HUMAN" and predicted == "HUMAN":
            human_correct += 1
        if true_label == "AI"    and predicted == "AI":
            ai_correct += 1
        logger.info("  [%2d] %-5s → score=%.1f (%s) %s",
                    i + 1, true_label, score,
                    predicted,
                    "✓" if is_correct else "✗")

    logger.info("\nRegression result: %d/30 correctly classified (binary, Mixed excluded)", correct)
    logger.info("  Human correct: %d/15 | AI correct: %d/15", human_correct, ai_correct)
    regression_pass = correct >= 24  # >= 80% pass threshold
    logger.info("✅ PASS (>= 80% accuracy)" if regression_pass else "❌ FAIL (< 80% accuracy)")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    logger.info("\n=== Phase F Summary ===")
    logger.info("Test 1 (Bar Council news)    : %s | C15 score=%.2f, verdict=%s",
                "PASS" if bar_council_pass else "FAIL", c15_score, c15_verdict)
    logger.info("Test 2 (Transformers wiki)   : %s | C15 score=%.2f, verdict=%s",
                "PASS" if transformer_pass else "FAIL", c15_score2, c15_verdict2)
    logger.info("Test 3 (HC3 regression/30)   : %s | %d/30 correct",
                "PASS" if regression_pass else "FAIL", correct)
    logger.info("")
    if bar_council_pass and transformer_pass and regression_pass:
        logger.info("✅ All Phase F validations PASSED.")
    else:
        logger.warning("⚠️  Some Phase F validations FAILED — review results above before deploying.")


if __name__ == "__main__":
    run_phase_f()
