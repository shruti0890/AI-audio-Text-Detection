"""
calibration/tune_fusion_weights.py

Correction 9  : Data-Driven Weight Rebalancing & Threshold Retuning.
Correction 12 : Percentile-derived thresholds (h_90 / ai_10).
Correction 13 : Threshold-Aware F1 Grid Search.
               - compute_thresholds() extracted as a single reusable function.
               - Curvature weight floor lowered from 0.40 → 0.20.
               - Every weight candidate is evaluated against its OWN
                 data-derived thresholds, not a fixed flat-50 cutoff.
               - Grid ranked by threshold-aware F1 (not ROC-AUC).
               - Top-5 saved with per-candidate thresholds.
               - Overfitting sanity check on 20 held-out samples.

1. Loads 120 tuning HC3 ground-truth samples (60 human, 60 ChatGPT AI).
2. Pre-calculates 4 calibrated signal sub-scores.
3. Performs grid search; curvature weight >= 0.20.
4. Maximizes threshold-aware F1 to find optimal weights.
5. Retunes classification decision thresholds from real distributions.
6. Runs overfitting check on 20 fresh HC3 held-out samples.
7. Saves configuration to fusion_config.json.
"""

from __future__ import annotations

import json
import logging
import sys
import numpy as np
from pathlib import Path

_CALIB_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _CALIB_DIR.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from text_forensics.pipeline import _load_baseline_stats
from text_forensics.signals.curvature import get_curvature
from text_forensics.signals.burstiness import get_burstiness
from text_forensics.signals.cliche_scanner import get_cliche_density
from text_forensics.signals.lexical_entropy import get_lexical_stats
from text_forensics.fusion import _cdf_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HUMAN_POOL_PATH = _CALIB_DIR / "hc3_test_pool_human.json"
AI_POOL_PATH    = _CALIB_DIR / "hc3_test_pool_ai.json"
CONFIG_OUT_PATH = _CALIB_DIR / "fusion_config.json"

# Number of samples reserved at the END of each pool file for overfitting check.
# These are never used during the 120-sample grid search.
HOLDOUT_PER_CLASS = 10   # 10 human + 10 AI = 20 total


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Calculate ROC-AUC without external sklearn dependency (Mann-Whitney U)."""
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    u = 0.0
    for p in pos:
        u += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(u / (n_pos * n_neg))


def compute_f1_matrix(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> dict:
    """Compute confusion matrix and F1-score at a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
    }


# ---------------------------------------------------------------------------
# Correction 12 / 13 — single reusable threshold function
# ---------------------------------------------------------------------------

def compute_thresholds(human_scores: np.ndarray, ai_scores: np.ndarray) -> tuple[float, float]:
    """
    Derive (t_human, t_ai) from real score distributions.

    Normal case (no overlap):
        t_human = 90th percentile of human scores
        t_ai    = 10th percentile of AI scores
        These form the edges of the uncertain/mixed band.

    Overlap case (h_90 >= ai_10):
        Distributions are too interleaved for a percentile-gap threshold.
        Falls back to a ±5-point buffer around the midpoint crossover and
        logs a WARNING — this indicates fundamental distribution ambiguity.

    Used in BOTH the per-candidate grid evaluation (Correction 13) and
    the final saved threshold (single source of truth, no duplicated logic).
    """
    h_90  = float(np.percentile(human_scores, 90))
    ai_10 = float(np.percentile(ai_scores,   10))

    if h_90 >= ai_10:
        crossover = (h_90 + ai_10) / 2.0
        t_human = round(crossover - 5.0, 2)
        t_ai    = round(crossover + 5.0, 2)
        logger.warning(
            "OVERLAP DETECTED: h_90 (%.2f) >= ai_10 (%.2f). "
            "No clean gap — falling back to crossover-centred band: "
            "t_human=%.2f, t_ai=%.2f. "
            "This indicates fundamental score-distribution ambiguity.",
            h_90, ai_10, t_human, t_ai,
        )
    else:
        t_human = h_90
        t_ai    = ai_10

    return t_human, t_ai


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def _extract_subscores(texts: list[str]) -> list[dict]:
    """Run pipeline signals and return calibrated sub-score dicts."""
    baseline  = _load_baseline_stats()
    _INVERTED = {"burstiness", "entropy"}
    result    = []

    for i, text in enumerate(texts):
        if (i + 1) % 20 == 0:
            logger.info("  Processing sample [%d/%d] ...", i + 1, len(texts))

        curv_raw    = get_curvature(text)
        burst_raw   = get_burstiness(text)
        cliche_raw  = get_cliche_density(text)
        entropy_raw = get_lexical_stats(text)["entropy"]

        raw_map = {
            "curvature": curv_raw,
            "burstiness": burst_raw,
            "cliche": cliche_raw,
            "entropy": entropy_raw,
        }

        sample_sub: dict = {}
        for k in ["curvature", "burstiness", "cliche", "entropy"]:
            b_key = "cliche_density" if k == "cliche" else k
            r_val = raw_map[k]
            if r_val is None:
                sample_sub[k] = None
            else:
                stats  = baseline.get(b_key, {})
                mu0    = stats.get("mu0",    0.0)
                sig0   = stats.get("sigma0", 1.0)
                cdf_val = _cdf_score(r_val, mu0, sig0)
                calib   = (100.0 - cdf_val) if k in _INVERTED else cdf_val
                sample_sub[k] = round(max(0.0, min(100.0, calib)), 2)

        result.append(sample_sub)

    return result


def precompute_sample_subscores() -> tuple[list[dict], np.ndarray, list[dict], np.ndarray]:
    """
    Returns:
        tuning_subscores (120 samples), tuning_y_true
        holdout_subscores (20 samples), holdout_y_true
    The last HOLDOUT_PER_CLASS entries of each pool file are reserved as
    the overfitting check set and are NOT included in the 120 tuning samples.
    """
    with open(HUMAN_POOL_PATH, encoding="utf-8") as f:
        all_human = json.load(f)
    with open(AI_POOL_PATH, encoding="utf-8") as f:
        all_ai = json.load(f)

    # Split: last HOLDOUT_PER_CLASS per class → holdout; remainder → tuning
    human_tune, human_hold = all_human[:-HOLDOUT_PER_CLASS], all_human[-HOLDOUT_PER_CLASS:]
    ai_tune,    ai_hold    = all_ai[:-HOLDOUT_PER_CLASS],    all_ai[-HOLDOUT_PER_CLASS:]

    tune_texts = human_tune + ai_tune
    hold_texts = human_hold + ai_hold

    tune_y = np.array([0] * len(human_tune) + [1] * len(ai_tune))
    hold_y = np.array([0] * len(human_hold) + [1] * len(ai_hold))

    logger.info(
        "Tuning set : %d samples (%d human, %d AI)",
        len(tune_texts), len(human_tune), len(ai_tune),
    )
    logger.info(
        "Holdout set: %d samples (%d human, %d AI) — reserved for overfitting check",
        len(hold_texts), len(human_hold), len(ai_hold),
    )

    logger.info("Extracting signals for tuning set (%d samples)...", len(tune_texts))
    tune_sub = _extract_subscores(tune_texts)

    logger.info("Extracting signals for holdout set (%d samples)...", len(hold_texts))
    hold_sub = _extract_subscores(hold_texts)

    return tune_sub, tune_y, hold_sub, hold_y


# ---------------------------------------------------------------------------
# Grid search — Correction 13: threshold-aware F1
# ---------------------------------------------------------------------------

def run_grid_search(
    subscores_list: list[dict],
    y_true: np.ndarray,
    holdout_subscores: list[dict],
    holdout_y: np.ndarray,
):
    """
    Grid search with threshold-aware F1 (Correction 13).

    For each weight combination:
      1. Compute fused scores for all 120 tuning samples.
      2. Call compute_thresholds() on THAT combination's human/AI score split.
      3. Use t_ai as the binary classification cutoff for F1
         (anything >= t_ai → "AI", anything < t_ai → "not AI").
         NOTE: this is a binary simplification — the 3-way band
         (Human / Mixed / AI) requires two thresholds, but F1 needs one.
         Using t_ai as the cutoff maximises precision on the AI class,
         which is the safety-critical direction.
      4. Rank by threshold-aware F1 DESC, then ROC-AUC DESC as tiebreaker.
    """
    logger.info("=== Correction 13: Threshold-Aware Grid Search ===")
    logger.info("Curvature floor: 0.20 (lowered from 0.40)")
    logger.info("F1 metric: binary at each candidate's own t_ai threshold")

    # Original pre-Correction-9 baseline for reference
    baseline_w = {"curvature": 0.40, "burstiness": 0.20, "cliche": 0.15, "entropy": 0.25}

    def compute_fused(weights_dict: dict) -> np.ndarray:
        scores = []
        for item in subscores_list:
            avail_w: dict = {}
            sub_vals: dict = {}
            for k in ["curvature", "burstiness", "cliche", "entropy"]:
                val = item[k]
                if val is not None:
                    avail_w[k]   = weights_dict[k]
                    sub_vals[k]  = val
            tot_w = sum(avail_w.values())
            if tot_w <= 0:
                scores.append(50.0)
            else:
                fused = sum((avail_w[k] / tot_w) * sub_vals[k] for k in avail_w)
                scores.append(fused)
        return np.array(scores)

    def compute_fused_from_sub(sub: list[dict], weights_dict: dict) -> np.ndarray:
        scores = []
        for item in sub:
            avail_w: dict = {}
            sub_vals: dict = {}
            for k in ["curvature", "burstiness", "cliche", "entropy"]:
                val = item[k]
                if val is not None:
                    avail_w[k]  = weights_dict[k]
                    sub_vals[k] = val
            tot_w = sum(avail_w.values())
            if tot_w <= 0:
                scores.append(50.0)
            else:
                fused = sum((avail_w[k] / tot_w) * sub_vals[k] for k in avail_w)
                scores.append(fused)
        return np.array(scores)

    # ---- Baseline reference ----
    base_scores = compute_fused(baseline_w)
    base_auc    = compute_roc_auc(y_true, base_scores)
    base_th, base_tai = compute_thresholds(
        base_scores[y_true == 0], base_scores[y_true == 1]
    )
    base_f1 = compute_f1_matrix(y_true, base_scores, threshold=base_tai)["f1"]
    logger.info(
        "Baseline weights %s → ROC-AUC: %.4f | t_ai=%.2f | F1(t_ai): %.4f",
        baseline_w, base_auc, base_tai, base_f1,
    )

    # ---- Grid search ----
    step       = 0.05
    candidates = []

    # Correction 13: curvature floor lowered to 0.20
    for wc in np.arange(0.20, 0.90, step):
        for wb in np.arange(0.05, 0.55, step):
            for wcl in np.arange(0.05, 0.45, step):
                we = round(1.0 - (wc + wb + wcl), 2)
                if 0.0 <= we <= 0.30:
                    w_dict = {
                        "curvature":  round(float(wc),  2),
                        "burstiness": round(float(wb),  2),
                        "cliche":     round(float(wcl), 2),
                        "entropy":    round(float(we),  2),
                    }
                    scores = compute_fused(w_dict)
                    auc    = compute_roc_auc(y_true, scores)

                    # Threshold-aware F1: derive thresholds from THIS candidate's
                    # own score distribution, then evaluate binary F1 at t_ai.
                    t_h, t_a = compute_thresholds(
                        scores[y_true == 0], scores[y_true == 1]
                    )
                    f1_ta = compute_f1_matrix(y_true, scores, threshold=t_a)["f1"]

                    candidates.append({
                        "weights":    w_dict,
                        "roc_auc":    round(auc,   4),
                        "f1_at_t_ai": round(f1_ta, 4),
                        "t_human":    round(t_h,   4),
                        "t_ai":       round(t_a,   4),
                        "scores":     scores,
                    })

    # Rank by threshold-aware F1 DESC, ROC-AUC DESC as tiebreaker
    candidates.sort(key=lambda x: (x["f1_at_t_ai"], x["roc_auc"]), reverse=True)

    logger.info("Grid search evaluated %d valid weight combinations.", len(candidates))
    logger.info("\n--- TOP 5 CANDIDATES (ranked by threshold-aware F1) ---")
    for idx, c in enumerate(candidates[:5], 1):
        logger.info(
            "%d. Weights: %s | t_human=%.2f t_ai=%.2f | ROC-AUC: %.4f | F1(t_ai): %.4f",
            idx, c["weights"], c["t_human"], c["t_ai"], c["roc_auc"], c["f1_at_t_ai"],
        )

    best         = candidates[0]
    best_weights = best["weights"]
    best_scores  = best["scores"]

    # ---- Correction 12 threshold logic — via shared function ----
    logger.info("\n=== Step 2: Final Threshold Derivation (Correction 12 logic) ===")
    human_scores = best_scores[y_true == 0]
    ai_scores    = best_scores[y_true == 1]

    h_75  = float(np.percentile(human_scores, 75))
    h_90  = float(np.percentile(human_scores, 90))
    ai_10 = float(np.percentile(ai_scores,   10))
    ai_25 = float(np.percentile(ai_scores,   25))

    logger.info(
        "Human Scores Stats: min=%.2f, median=%.2f, 75th=%.2f, 90th=%.2f, max=%.2f",
        np.min(human_scores), np.median(human_scores), h_75, h_90, np.max(human_scores),
    )
    logger.info(
        "AI Scores Stats:    min=%.2f, 10th=%.2f, 25th=%.2f, median=%.2f, max=%.2f",
        np.min(ai_scores), ai_10, ai_25, np.median(ai_scores), np.max(ai_scores),
    )

    # Single call to the shared function — no duplicated logic
    t_human, t_ai = compute_thresholds(human_scores, ai_scores)

    band_width = round(t_ai - t_human, 2)
    logger.info(
        "Final thresholds: t_human=%.2f, t_ai=%.2f | Band width: %.2f pts%s",
        t_human, t_ai, band_width,
        " [NARROW — good separation]" if band_width <= 15
        else " [WIDE (>15 pts) — significant distribution overlap]",
    )

    # ---- Old-vs-new comparison (Correction 12 reference) ----
    C12_T_HUMAN = 76.52  # Correction 12 result with old 75/5/20/0 weights
    C12_T_AI    = 86.50

    def classify_band(scores: np.ndarray, th: float, ta: float) -> np.ndarray:
        return np.where(scores < th, "human", np.where(scores <= ta, "mixed", "ai"))

    # Correction 12 config on tuning set (for direct comparison)
    c12_scores   = compute_fused_from_sub(subscores_list, {"curvature": 0.75, "burstiness": 0.05, "cliche": 0.20, "entropy": 0.00})
    c12_verdicts = classify_band(c12_scores, C12_T_HUMAN, C12_T_AI)
    new_verdicts = classify_band(best_scores, t_human, t_ai)

    logger.info(
        "\n--- Correction 12 config (75/5/20/0, t=%.2f/%.2f) on %d samples ---"
        "\n  Human: %d | Mixed: %d | AI: %d",
        C12_T_HUMAN, C12_T_AI, len(c12_verdicts),
        np.sum(c12_verdicts == "human"),
        np.sum(c12_verdicts == "mixed"),
        np.sum(c12_verdicts == "ai"),
    )
    logger.info(
        "\n--- Correction 13 config (%s, t=%.2f/%.2f) on %d samples ---"
        "\n  Human: %d | Mixed: %d | AI: %d",
        best_weights, t_human, t_ai, len(new_verdicts),
        np.sum(new_verdicts == "human"),
        np.sum(new_verdicts == "mixed"),
        np.sum(new_verdicts == "ai"),
    )

    human_verdicts = new_verdicts[:len(human_scores)]
    ai_verdicts    = new_verdicts[len(human_scores):]
    logger.info(
        "Ground-truth Human (%d) under NEW thresholds: %d Human, %d Mixed, %d AI",
        len(human_scores),
        np.sum(human_verdicts == "human"),
        np.sum(human_verdicts == "mixed"),
        np.sum(human_verdicts == "ai"),
    )
    logger.info(
        "Ground-truth AI    (%d) under NEW thresholds: %d AI,    %d Mixed, %d Human",
        len(ai_scores),
        np.sum(ai_verdicts == "ai"),
        np.sum(ai_verdicts == "mixed"),
        np.sum(ai_verdicts == "human"),
    )

    # ---- Step 4: Overfitting sanity check ----
    logger.info("\n=== Step 4: Overfitting Sanity Check (20 held-out samples) ===")
    hold_scores = compute_fused_from_sub(holdout_subscores, best_weights)
    hold_auc    = compute_roc_auc(holdout_y, hold_scores)
    # Use the same thresholds derived from training data
    hold_f1_d   = compute_f1_matrix(holdout_y, hold_scores, threshold=t_ai)
    hold_f1     = hold_f1_d["f1"]

    logger.info(
        "Holdout (20 samples): ROC-AUC=%.4f | F1(t_ai=%.2f)=%.4f",
        hold_auc, t_ai, hold_f1,
    )
    tuning_f1 = best["f1_at_t_ai"]
    drop       = round(tuning_f1 - hold_f1, 4)
    if drop > 0.10:
        logger.warning(
            "OVERFITTING RISK: holdout F1 (%.4f) is %.4f below tuning F1 (%.4f). "
            "Weights may be overfitted to the 120-sample HC3 tuning set.",
            hold_f1, drop, tuning_f1,
        )
    elif drop > 0.05:
        logger.warning(
            "MILD OVERFITTING SIGNAL: holdout F1 drop = %.4f (threshold 0.05). "
            "Monitor on a larger validation set before production use.",
            drop,
        )
    else:
        logger.info(
            "Overfitting check PASSED: holdout F1 drop = %.4f (within 0.05 tolerance).",
            drop,
        )
    logger.info(
        "Confusion matrix (holdout): TP=%d FP=%d TN=%d FN=%d | Precision=%.4f Recall=%.4f",
        hold_f1_d["tp"], hold_f1_d["fp"], hold_f1_d["tn"], hold_f1_d["fn"],
        hold_f1_d["precision"], hold_f1_d["recall"],
    )

    # ---- Save ----
    config_data = {
        "weights": best_weights,
        "thresholds": {
            "human_max":   t_human,
            "mixed_range": [t_human, t_ai],
            "ai_min":      t_ai,
        },
        "tuned_on":         "hc3_test_pool (tuning set: 100 samples per class from pool, ~120 total after split)",
        "tuned_date":       "2026-08-14",
        "threshold_method": (
            "Correction 12/13 — percentile-derived: t_human=h_90, t_ai=ai_10; "
            "overlap fallback: crossover±5"
        ),
        "selection_criterion": (
            "Correction 13 — threshold-aware F1 (binary at t_ai). "
            "F1 evaluated at each candidate's own data-derived t_ai, not flat-50. "
            "Ranked by F1 DESC, ROC-AUC DESC as tiebreaker."
        ),
        "baseline_comparison": {
            "old_weights":  {"curvature": 0.40, "burstiness": 0.20, "cliche": 0.15, "entropy": 0.25},
            "old_roc_auc":  round(base_auc, 4),
            "old_f1_flat50": round(base_f1, 4),
            "new_weights":  best_weights,
            "new_roc_auc":  best["roc_auc"],
            "new_f1_at_tai": best["f1_at_t_ai"],
        },
        "overfitting_check": {
            "holdout_n":         len(holdout_subscores),
            "holdout_roc_auc":   round(hold_auc,  4),
            "holdout_f1_at_tai": round(hold_f1,   4),
            "tuning_f1_at_tai":  round(tuning_f1, 4),
            "f1_drop":           round(drop,       4),
            "verdict": (
                "OVERFITTING RISK"        if drop > 0.10 else
                "MILD OVERFITTING SIGNAL" if drop > 0.05 else
                "PASSED"
            ),
        },
        "top_5_grid_candidates": [
            {
                "weights":    c["weights"],
                "t_human":    c["t_human"],
                "t_ai":       c["t_ai"],
                "roc_auc":    c["roc_auc"],
                "f1_at_t_ai": c["f1_at_t_ai"],
            }
            for c in candidates[:5]
        ],
    }

    with open(CONFIG_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    logger.info("Saved Correction 13 config to %s", CONFIG_OUT_PATH)
    return config_data


if __name__ == "__main__":
    tune_sub, tune_y, hold_sub, hold_y = precompute_sample_subscores()
    run_grid_search(tune_sub, tune_y, hold_sub, hold_y)
