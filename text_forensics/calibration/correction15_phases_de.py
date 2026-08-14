"""
Correction 15 — Phases D + E: Genre-Diverse Grid Search + Threshold Re-Derivation.

Phase D: Re-runs the threshold-aware F1 grid search (Correction 13's method) on a
         genre-balanced held-out set built from:
           - Original HC3 test pool (hc3_test_pool_human/ai.json) — conversational genre
           - Phase A/B holdout splits (25/genre/side from genre_corpus_additions_*.json)
         This held-out set was NEVER used in Phase C baseline stats computation.

Phase E: Re-derives human_max / ai_min using Correction 12's percentile method on the
         winning Phase D weights, then saves the updated fusion_config.json.

Comparison reported: new genre-diverse ROC-AUC/F1 vs Correction 13 HC3-only numbers.

Run from project root:
    python text_forensics/calibration/correction15_phases_de.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date
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
from text_forensics.calibration.tune_fusion_weights import (
    compute_roc_auc, compute_f1_matrix, compute_thresholds
)

CONFIG_OUT_PATH = _CALIB_DIR / "fusion_config.json"

# Correction 13 reference numbers for comparison
C13_ROC_AUC   = 1.0000
C13_F1_AT_TAI = 0.9474
C13_WEIGHTS   = {"curvature": 0.50, "burstiness": 0.15, "cliche": 0.25, "entropy": 0.10}
C14_WEIGHTS   = {"curvature": 0.5389, "burstiness": 0.1617, "cliche": 0.2694, "entropy": 0.03}


# ---------------------------------------------------------------------------
# Signal extraction (same logic as tune_fusion_weights.py)
# ---------------------------------------------------------------------------

def _extract_subscores(texts: list[str]) -> list[dict]:
    """Run all 4 signals and return calibrated sub-score dicts."""
    baseline  = _load_baseline_stats()
    _INVERTED = {"burstiness", "entropy"}
    result    = []

    for i, text in enumerate(texts):
        if (i + 1) % 20 == 0:
            logger.info("  [%d/%d] sub-score extraction...", i + 1, len(texts))

        curv_raw    = get_curvature(text)
        burst_raw   = get_burstiness(text)
        cliche_raw  = get_cliche_density(text)
        entropy_raw = get_lexical_stats(text)["entropy"]

        raw_map = {
            "curvature":  curv_raw,
            "burstiness": burst_raw,
            "cliche":     cliche_raw,
            "entropy":    entropy_raw,
        }

        sample_sub: dict = {}
        for k in ["curvature", "burstiness", "cliche", "entropy"]:
            b_key = "cliche_density" if k == "cliche" else k
            r_val = raw_map[k]
            if r_val is None:
                sample_sub[k] = None
            else:
                stats   = baseline.get(b_key, {})
                mu0     = stats.get("mu0",    0.0)
                sig0    = stats.get("sigma0", 1.0)
                cdf_val = _cdf_score(r_val, mu0, sig0)
                calib   = (100.0 - cdf_val) if k in _INVERTED else cdf_val
                sample_sub[k] = round(max(0.0, min(100.0, calib)), 2)

        result.append(sample_sub)

    return result


def _fuse(subscores_list: list[dict], weights: dict) -> np.ndarray:
    """Compute fused scores from sub-score list and weights dict."""
    scores = []
    for item in subscores_list:
        avail_w: dict = {}
        sub_vals: dict = {}
        for k in ["curvature", "burstiness", "cliche", "entropy"]:
            val = item[k]
            if val is not None:
                avail_w[k]  = weights[k]
                sub_vals[k] = val
        tot_w = sum(avail_w.values())
        if tot_w <= 0:
            scores.append(50.0)
        else:
            fused = sum((avail_w[k] / tot_w) * sub_vals[k] for k in avail_w)
            scores.append(fused)
    return np.array(scores)


# ---------------------------------------------------------------------------
# Build genre-diverse held-out set
# ---------------------------------------------------------------------------

def build_holdout_set() -> tuple[list[str], np.ndarray, dict]:
    """
    Build the genre-balanced held-out set for Phase D grid search.

    Composition:
      Human side:
        - 50 from HC3 test pool (conversational — the full hc3_test_pool_human.json)
        - 25 from each new genre (holdout split from genre_corpus_additions_human.json)
      AI side:
        - 50 from HC3 test pool (conversational — the full hc3_test_pool_ai.json)
        - 25 from each new genre (holdout split from genre_corpus_additions_ai.json)

    Returns: (texts, y_true, composition_report)
    """
    # HC3 test pool
    with open(_CALIB_DIR / "hc3_test_pool_human.json", encoding="utf-8") as f:
        hc3_human = json.load(f)
    with open(_CALIB_DIR / "hc3_test_pool_ai.json", encoding="utf-8") as f:
        hc3_ai = json.load(f)

    # Genre additions (holdout split only)
    human_additions: list[str] = []
    genre_human_counts: dict   = {}
    if (_CALIB_DIR / "genre_corpus_additions_human.json").exists():
        with open(_CALIB_DIR / "genre_corpus_additions_human.json", encoding="utf-8") as f:
            for entry in json.load(f):
                if entry.get("split") == "holdout":
                    human_additions.append(entry["text"])
                    g = entry.get("genre", "unknown")
                    genre_human_counts[g] = genre_human_counts.get(g, 0) + 1
    else:
        logger.warning("genre_corpus_additions_human.json not found. Using HC3-only held-out set.")

    ai_additions: list[str] = []
    genre_ai_counts: dict   = {}
    if (_CALIB_DIR / "genre_corpus_additions_ai.json").exists():
        with open(_CALIB_DIR / "genre_corpus_additions_ai.json", encoding="utf-8") as f:
            for entry in json.load(f):
                if entry.get("split") == "holdout":
                    ai_additions.append(entry["text"])
                    g = entry.get("genre", "unknown")
                    genre_ai_counts[g] = genre_ai_counts.get(g, 0) + 1
    else:
        logger.warning("genre_corpus_additions_ai.json not found. Using HC3-only held-out set.")

    # Cap HC3 to 50 per class for genre balance (genres provide 25 each × 3 = 75 more per side)
    hc3_h = hc3_human[:50]
    hc3_a = hc3_ai[:50]

    all_human = hc3_h + human_additions
    all_ai    = hc3_a + ai_additions
    texts     = all_human + all_ai
    y_true    = np.array([0] * len(all_human) + [1] * len(all_ai))

    composition = {
        "human": {
            "hc3_conversational": len(hc3_h),
            **genre_human_counts,
            "total": len(all_human),
        },
        "ai": {
            "hc3_conversational": len(hc3_a),
            **genre_ai_counts,
            "total": len(all_ai),
        },
        "total": len(texts),
    }

    logger.info("Held-out set composition:")
    logger.info("  Human: %d total — HC3 conv=%d | genre=%s",
                len(all_human), len(hc3_h), genre_human_counts)
    logger.info("  AI:    %d total — HC3 conv=%d | genre=%s",
                len(all_ai),    len(hc3_a), genre_ai_counts)

    return texts, y_true, composition


# ---------------------------------------------------------------------------
# Phase D: Grid search
# ---------------------------------------------------------------------------

def run_phase_d(subscores: list[dict], y_true: np.ndarray) -> tuple[dict, dict]:
    """
    Run threshold-aware F1 grid search on genre-diverse held-out set.
    Returns (best_candidate, all_top5).
    """
    logger.info("\n=== Phase D: Genre-Diverse Threshold-Aware Grid Search ===")
    logger.info("Curvature floor: 0.20 | F1 evaluated at each candidate's own t_ai")

    step       = 0.05
    candidates = []

    for wc in np.arange(0.20, 0.90, step):
        for wb in np.arange(0.05, 0.55, step):
            for wcl in np.arange(0.05, 0.45, step):
                we = round(1.0 - (round(float(wc), 2) + round(float(wb), 2) + round(float(wcl), 2)), 2)
                if 0.0 <= we <= 0.30:
                    w_dict = {
                        "curvature":  round(float(wc),  2),
                        "burstiness": round(float(wb),  2),
                        "cliche":     round(float(wcl), 2),
                        "entropy":    round(float(we),  2),
                    }
                    scores = _fuse(subscores, w_dict)
                    auc    = compute_roc_auc(y_true, scores)
                    t_h, t_a = compute_thresholds(scores[y_true == 0], scores[y_true == 1])
                    f1_ta    = compute_f1_matrix(y_true, scores, threshold=t_a)["f1"]
                    candidates.append({
                        "weights":    w_dict,
                        "roc_auc":    round(auc,   4),
                        "f1_at_t_ai": round(f1_ta, 4),
                        "t_human":    round(t_h,   4),
                        "t_ai":       round(t_a,   4),
                        "scores":     scores,
                    })

    candidates.sort(key=lambda x: (x["f1_at_t_ai"], x["roc_auc"]), reverse=True)
    logger.info("Evaluated %d weight combinations.", len(candidates))

    # Reference: C13 and C14 weights on THIS held-out set
    for label, w in [("C13 (0.50/0.15/0.25/0.10)", C13_WEIGHTS),
                     ("C14 (0.5389/0.1617/0.2694/0.03)", C14_WEIGHTS)]:
        sc  = _fuse(subscores, w)
        auc = compute_roc_auc(y_true, sc)
        th, ta = compute_thresholds(sc[y_true == 0], sc[y_true == 1])
        f1  = compute_f1_matrix(y_true, sc, threshold=ta)["f1"]
        logger.info("Reference %s: ROC-AUC=%.4f | t_ai=%.2f | F1=%.4f", label, auc, ta, f1)

    logger.info("\n--- TOP 5 CANDIDATES (genre-diverse held-out) ---")
    for i, c in enumerate(candidates[:5], 1):
        logger.info(
            "%d. Weights=%s | t_human=%.2f t_ai=%.2f | ROC-AUC=%.4f | F1=%.4f",
            i, c["weights"], c["t_human"], c["t_ai"], c["roc_auc"], c["f1_at_t_ai"]
        )

    # Choose winner: highest F1, then ROC-AUC
    best = candidates[0]
    logger.info("\nWINNER: %s | t_human=%.2f | t_ai=%.2f | ROC-AUC=%.4f | F1=%.4f",
                best["weights"], best["t_human"], best["t_ai"],
                best["roc_auc"], best["f1_at_t_ai"])

    top5 = [
        {"weights": c["weights"], "t_human": c["t_human"], "t_ai": c["t_ai"],
         "roc_auc": c["roc_auc"], "f1_at_t_ai": c["f1_at_t_ai"]}
        for c in candidates[:5]
    ]
    best_clean = {k: v for k, v in best.items() if k != "scores"}
    return best_clean, top5


# ---------------------------------------------------------------------------
# Phase E: Threshold re-derivation + save config
# ---------------------------------------------------------------------------

def run_phase_e(
    best: dict,
    top5: list[dict],
    subscores: list[dict],
    y_true: np.ndarray,
    composition: dict,
) -> None:
    """Re-derive thresholds from best weights and save fusion_config.json."""
    logger.info("\n=== Phase E: Threshold Re-Derivation + fusion_config.json Update ===")

    best_w  = best["weights"]
    scores  = _fuse(subscores, best_w)
    h_scores = scores[y_true == 0]
    a_scores = scores[y_true == 1]

    t_human, t_ai = compute_thresholds(h_scores, a_scores)
    band_width     = round(t_ai - t_human, 2)

    logger.info("Thresholds (Correction 15, genre-diverse):")
    logger.info("  t_human (h_90) = %.4f", t_human)
    logger.info("  t_ai    (ai_10)= %.4f", t_ai)
    logger.info("  Band width     = %.2f pts   (C12/13 band was 7.71 pts)", band_width)

    # Load old config for baseline_comparison
    old_config: dict = {}
    if CONFIG_OUT_PATH.exists():
        with open(CONFIG_OUT_PATH, encoding="utf-8") as f:
            old_config = json.load(f)

    config_data = {
        "weights": best_w,
        "thresholds": {
            "human_max":   t_human,
            "mixed_range": [t_human, t_ai],
            "ai_min":      t_ai,
        },
        "tuned_on": (
            "Correction 15 genre-diverse held-out set: "
            f"HC3 conversational ({composition['human']['hc3_conversational']} human "
            f"+ {composition['ai']['hc3_conversational']} AI) "
            f"+ news/legal/technical genre additions "
            f"({composition['human'].get('news', 0) + composition['human'].get('legal', 0) + composition['human'].get('technical', 0)} human "
            f"+ {composition['ai'].get('news', 0) + composition['ai'].get('legal', 0) + composition['ai'].get('technical', 0)} AI). "
            f"Total held-out: {composition['total']} samples."
        ),
        "tuned_date": date.today().isoformat(),
        "threshold_method": (
            "Correction 15 — same percentile method as Corrections 12/13: "
            "t_human=h_90, t_ai=ai_10; overlap fallback: crossover±5. "
            "Applied to genre-diverse held-out set (conversational + news + legal + technical)."
        ),
        "selection_criterion": (
            "Correction 15 — threshold-aware F1 grid search on genre-diverse held-out set. "
            "Same Correction 13 algorithm, new genre-balanced evaluation corpus. "
            "Curvature floor 0.20. Ranked by F1 DESC, ROC-AUC DESC as tiebreaker."
        ),
        "calibration_corpus_note": (
            "Correction 15: baseline_stats.json (mu0/sigma0) recomputed on ~475 samples: "
            "250 HC3 human_answers + 75 CNN/DailyMail news + 75 BillSum legal + 75 Wikitext-103 technical. "
            "Curvature uses the full passage with no token ceiling (distilgpt2 context window = 1024 tokens). "
            "Genre coverage: conversational, news, legal/bureaucratic, technical/informational. "
            "Known gaps: social media, creative writing, poetry — not covered in this calibration."
        ),
        "supersedes": "Corrections 12, 13, 14",
        "manual_override_note_c14": (
            "[Retained for audit trail] C14 entropy cap at 0.03 is superseded by C15 genre-diverse re-tune. "
            "The C14 cap failed to fix the Bar Council article (curvature was the root cause, not entropy). "
            "C15 addresses root cause via genre-diverse calibration."
        ),
        "baseline_comparison": {
            "c13_hc3_only": {
                "weights":    C13_WEIGHTS,
                "roc_auc":    C13_ROC_AUC,
                "f1_at_tai":  C13_F1_AT_TAI,
                "t_human":    66.131,
                "t_ai":       73.838,
            },
            "c15_genre_diverse": {
                "weights":    best_w,
                "roc_auc":    best["roc_auc"],
                "f1_at_tai":  best["f1_at_t_ai"],
                "t_human":    t_human,
                "t_ai":       t_ai,
            },
        },
        "top_5_grid_candidates": top5,
    }

    with open(CONFIG_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    logger.info("Saved Correction 15 config to %s", CONFIG_OUT_PATH)
    logger.info("Band width: %.2f pts (C13 was 7.71 pts)", band_width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== Correction 15: Phases D + E ===")

    texts, y_true, composition = build_holdout_set()
    logger.info("Extracting sub-scores for %d held-out samples...", len(texts))
    subscores = _extract_subscores(texts)
    logger.info("Sub-score extraction complete.")

    best, top5 = run_phase_d(subscores, y_true)
    run_phase_e(best, top5, subscores, y_true, composition)

    logger.info("\n=== Phases D + E Complete ===")


if __name__ == "__main__":
    main()
