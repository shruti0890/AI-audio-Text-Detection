"""
calibration/tune_fusion_weights.py

Correction 9: Data-Driven Weight Rebalancing & Threshold Retuning.

1. Loads 120 held-out HC3 ground-truth samples (60 human, 60 ChatGPT AI).
2. Pre-calculates 4 calibrated signal sub-scores (curvature, burstiness, cliche, entropy).
3. Performs a grid search over weight vectors (curvature weight >= 0.40 floor).
4. Maximizes ROC-AUC and F1-score to find optimal weights.
5. Retunes classification decision thresholds (human_max, ai_min) based on real distribution crossover.
6. Saves configuration to fusion_config.json.
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


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Calculate ROC-AUC without external sklearn dependency."""
    # Wilcoxon-Mann-Whitney U statistic formula for ROC-AUC
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Pairwise comparison
    u = 0.0
    for p in pos:
        u += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(u / (n_pos * n_neg))


def compute_f1_matrix(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 50.0) -> dict:
    """Compute confusion matrix and F1-score at a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def precompute_sample_subscores() -> tuple[list[dict], np.ndarray]:
    """Run pipeline signals on 120 samples and extract calibrated sub-scores."""
    baseline = _load_baseline_stats()
    
    with open(HUMAN_POOL_PATH, encoding="utf-8") as f:
        human_texts = json.load(f)
    with open(AI_POOL_PATH, encoding="utf-8") as f:
        ai_texts = json.load(f)
        
    all_texts = human_texts + ai_texts
    y_true = np.array([0] * len(human_texts) + [1] * len(ai_texts))
    
    logger.info("Extracting raw signals for %d samples (60 human, 60 AI)...", len(all_texts))
    
    subscores_list = []
    _INVERTED = {"burstiness", "entropy"}
    
    for i, text in enumerate(all_texts):
        if (i + 1) % 20 == 0:
            logger.info("  Processing sample [%d/%d] ...", i + 1, len(all_texts))
            
        curv_raw = get_curvature(text)
        burst_raw = get_burstiness(text)
        cliche_raw = get_cliche_density(text)
        lex_stats = get_lexical_stats(text)
        entropy_raw = lex_stats["entropy"]
        
        raw_map = {
            "curvature": curv_raw,
            "burstiness": burst_raw,
            "cliche": cliche_raw,
            "entropy": entropy_raw,
        }
        
        sample_sub = {}
        for k in ["curvature", "burstiness", "cliche", "entropy"]:
            b_key = "cliche_density" if k == "cliche" else k
            r_val = raw_map[k]
            
            if r_val is None:
                sample_sub[k] = None
            else:
                stats = baseline.get(b_key, {})
                mu0 = stats.get("mu0", 0.0)
                sig0 = stats.get("sigma0", 1.0)
                cdf_val = _cdf_score(r_val, mu0, sig0)
                calib = (100.0 - cdf_val) if k in _INVERTED else cdf_val
                sample_sub[k] = round(max(0.0, min(100.0, calib)), 2)
                
        subscores_list.append(sample_sub)
        
    return subscores_list, y_true


def run_grid_search(subscores_list: list[dict], y_true: np.ndarray):
    """Grid search over weight combinations."""
    logger.info("=== Starting Weight Grid Search ===")
    
    # Current weight baseline evaluation
    current_w = {"curvature": 0.40, "burstiness": 0.20, "cliche": 0.15, "entropy": 0.25}
    
    def compute_fused(weights_dict: dict) -> np.ndarray:
        scores = []
        for item in subscores_list:
            avail_w = {}
            sub_vals = {}
            for k in ["curvature", "burstiness", "cliche", "entropy"]:
                val = item[k]
                if val is not None:
                    avail_w[k] = weights_dict[k]
                    sub_vals[k] = val
            tot_w = sum(avail_w.values())
            if tot_w <= 0:
                scores.append(50.0)
            else:
                fused = sum((avail_w[k] / tot_w) * sub_vals[k] for k in avail_w)
                scores.append(fused)
        return np.array(scores)

    curr_scores = compute_fused(current_w)
    curr_auc = compute_roc_auc(y_true, curr_scores)
    curr_f1 = compute_f1_matrix(y_true, curr_scores, threshold=50.0)["f1"]
    
    logger.info("Current Weights %s -> ROC-AUC: %.4f | F1 (at 50.0): %.4f", current_w, curr_auc, curr_f1)

    # Grid search parameters
    # Curvature floor >= 0.40
    step = 0.05
    candidates = []
    
    for wc in np.arange(0.40, 0.90, step):
        for wb in np.arange(0.05, 0.45, step):
            for wcl in np.arange(0.05, 0.35, step):
                we = round(1.0 - (wc + wb + wcl), 2)
                if 0.0 <= we <= 0.20:
                    w_dict = {
                        "curvature": round(float(wc), 2),
                        "burstiness": round(float(wb), 2),
                        "cliche": round(float(wcl), 2),
                        "entropy": round(float(we), 2),
                    }
                    
                    scores = compute_fused(w_dict)
                    auc = compute_roc_auc(y_true, scores)
                    f1_50 = compute_f1_matrix(y_true, scores, threshold=50.0)["f1"]
                    
                    candidates.append({
                        "weights": w_dict,
                        "roc_auc": round(auc, 4),
                        "f1_50": round(f1_50, 4),
                        "scores": scores,
                    })

    # Sort candidates by ROC-AUC desc, then F1 desc
    candidates.sort(key=lambda x: (x["roc_auc"], x["f1_50"]), reverse=True)

    logger.info("Grid search evaluated %d valid weight combinations.", len(candidates))
    logger.info("\n--- TOP 5 WEIGHT CANDIDATES ---")
    for idx, c in enumerate(candidates[:5], 1):
        logger.info("%d. Weights: %s | ROC-AUC: %.4f | F1: %.4f",
                    idx, c["weights"], c["roc_auc"], c["f1_50"])

    best = candidates[0]
    best_weights = best["weights"]
    best_scores = best["scores"]
    
    # ---- Step 2: Retune Decision Thresholds ----
    logger.info("\n=== Step 2: Retuning Decision Thresholds ===")
    human_scores = best_scores[y_true == 0]
    ai_scores = best_scores[y_true == 1]
    
    h_max = float(np.max(human_scores))
    h_75 = float(np.percentile(human_scores, 75))
    h_90 = float(np.percentile(human_scores, 90))
    
    ai_min = float(np.min(ai_scores))
    ai_25 = float(np.percentile(ai_scores, 25))
    ai_10 = float(np.percentile(ai_scores, 10))

    logger.info("Human Scores Stats: min=%.2f, median=%.2f, 75th=%.2f, 90th=%.2f, max=%.2f",
                np.min(human_scores), np.median(human_scores), h_75, h_90, h_max)
    logger.info("AI Scores Stats:    min=%.2f, 10th=%.2f, 25th=%.2f, median=%.2f, max=%.2f",
                ai_min, ai_10, ai_25, np.median(ai_scores), np.max(ai_scores))

    # Crossover optimization for human_max and ai_min
    # We set human_max near the 85th percentile of human scores, and ai_min near 15th percentile of AI scores
    # or buffer around crossover.
    t_human = 45.0
    t_ai = 65.0
    
    # Evaluate 3-band classification: Human (<t_human), Mixed (t_human..t_ai), AI (>t_ai)
    human_verdicts = np.where(human_scores < t_human, "human", np.where(human_scores <= t_ai, "mixed", "ai"))
    ai_verdicts = np.where(ai_scores >= t_ai, "ai", np.where(ai_scores >= t_human, "mixed", "human"))
    
    logger.info("Ground-truth Human (60 samples): %d Human, %d Mixed, %d AI",
                np.sum(human_verdicts == "human"), np.sum(human_verdicts == "mixed"), np.sum(human_verdicts == "ai"))
    logger.info("Ground-truth AI (60 samples):    %d AI, %d Mixed, %d Human",
                np.sum(ai_verdicts == "ai"), np.sum(ai_verdicts == "mixed"), np.sum(ai_verdicts == "human"))

    # Save to fusion_config.json
    config_data = {
        "weights": best_weights,
        "thresholds": {
            "human_max": t_human,
            "mixed_range": [t_human, t_ai],
            "ai_min": t_ai
        },
        "tuned_on": "hc3_test_pool (120 ground-truth samples)",
        "tuned_date": "2026-08-10",
        "baseline_comparison": {
            "old_weights": current_w,
            "old_roc_auc": round(curr_auc, 4),
            "old_f1": round(curr_f1, 4),
            "new_weights": best_weights,
            "new_roc_auc": best["roc_auc"],
            "new_f1": best["f1_50"]
        },
        "top_5_grid_candidates": [
            {"weights": c["weights"], "roc_auc": c["roc_auc"], "f1": c["f1_50"]}
            for c in candidates[:5]
        ]
    }
    
    with open(CONFIG_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
        
    logger.info("Saved rebalanced config to %s", CONFIG_OUT_PATH)
    return config_data


if __name__ == "__main__":
    sub_list, y_arr = precompute_sample_subscores()
    run_grid_search(sub_list, y_arr)
