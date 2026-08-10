"""
tests/run_phase_1_8_hc3_validation.py

Correction 2: Phase 1.8 validation using REAL HC3 data (fast mode for validation).

Replaces the synthetic self-written AI samples with actual ChatGPT answers
from the HC3 dataset (held out from calibration in Correction 1).

Inputs:
  - text_forensics/calibration/hc3_test_pool_human.json (held-out human answers)
  - text_forensics/calibration/hc3_test_pool_ai.json    (held-out ChatGPT answers)

Output:
  - text_forensics/tests/results_phase_1_8.md (overwrites old synthetic-data results)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from text_forensics.signals.curvature import get_curvature
from text_forensics.signals.burstiness import get_burstiness
from text_forensics.signals.cliche_scanner import get_cliche_density
from text_forensics.signals.lexical_entropy import get_lexical_stats
from text_forensics.fusion import compute_text_score

_HUMAN_POOL = Path(__file__).parent.parent / "calibration" / "hc3_test_pool_human.json"
_AI_POOL    = Path(__file__).parent.parent / "calibration" / "hc3_test_pool_ai.json"
_STATS_FILE = Path(__file__).parent.parent / "calibration" / "baseline_stats.json"
_RESULTS_MD = Path(__file__).parent / "results_phase_1_8.md"

N_VALIDATE = 20   # 20 human + 20 AI = 40 total


def load_pools() -> tuple[list[str], list[str]]:
    with open(_HUMAN_POOL, encoding="utf-8") as f:
        human = json.load(f)
    with open(_AI_POOL, encoding="utf-8") as f:
        ai = json.load(f)
    return human[:N_VALIDATE], ai[:N_VALIDATE]


def analyze_text_fast(text: str, baseline_stats: dict) -> dict:
    """Fast scoring without T5 paraphrasing pass for validation loop."""
    lex = get_lexical_stats(text)
    curv = get_curvature(text)
    burst = get_burstiness(text)
    cliche = get_cliche_density(text)

    raw_signals = {
        "curvature_raw": curv,
        "burstiness_raw": burst,
        "cliche_density_raw": cliche,
        "entropy_raw": lex["entropy"],
    }
    fused = compute_text_score(raw_signals, baseline_stats)
    return {
        "text_score": fused["text_score"],
        "signals": {
            "curvature_raw": curv,
            "curvature_score": fused["sub_scores"].get("curvature_score"),
            "burstiness_raw": burst,
            "burstiness_score": fused["sub_scores"].get("burstiness_score"),
            "cliche_density_pct": cliche,
            "cliche_score": fused["sub_scores"].get("cliche_density_score"),
            "ttr": lex["ttr"],
            "entropy": lex["entropy"],
            "entropy_score": fused["sub_scores"].get("entropy_score"),
        },
        "stability_flag": "stable",
        "paraphrase_delta": 0.0,
        "compared_on_truncated": False,
    }


def run_hc3_validation() -> None:
    import numpy as np
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

    with open(_STATS_FILE, encoding="utf-8") as f:
        baseline_stats = json.load(f)

    human_samples, ai_samples = load_pools()
    all_samples = [(t, 0) for t in human_samples] + [(t, 1) for t in ai_samples]

    y_true: list[int] = []
    y_scores: list[float] = []
    y_pred: list[int] = []
    per_sample: list[dict] = []
    errors: list[str] = []

    print(f"Scoring {len(all_samples)} HC3 samples...")
    for i, (text, label) in enumerate(all_samples):
        kind = "HUMAN" if label == 0 else "AI"
        try:
            res = analyze_text_fast(text, baseline_stats)
            score = res["text_score"]
            pred = 1 if score >= 50 else 0
            y_true.append(label)
            y_scores.append(score)
            y_pred.append(pred)
            per_sample.append({
                "label": kind,
                "score": score,
                "pred": "AI" if pred == 1 else "Human",
                "correct": pred == label,
            })
        except Exception as e:
            errors.append(f"Sample {i+1} ({kind}): {e}")

    y_true_arr = np.array(y_true)
    y_scores_arr = np.array(y_scores)
    y_pred_arr = np.array(y_pred)

    auc = roc_auc_score(y_true_arr, y_scores_arr)
    f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
    cm = confusion_matrix(y_true_arr, y_pred_arr)

    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    edge_cases = [
        {
            "label": "Under 50 words",
            "text":  "The sun rose over the mountains. Birds began to sing. A light wind moved through the grass.",
        },
        {
            "label": "Half human / half AI (spliced)",
            "text":  (
                "My uncle repaired watches for thirty years in a shop no wider than a hallway. "
                "In today's rapidly evolving landscape, it is important to note that leveraging "
                "cutting-edge artificial intelligence is pivotal to driving transformative outcomes."
            ),
        },
    ]
    edge_results = []
    for ec in edge_cases:
        r = analyze_text_fast(ec["text"], baseline_stats)
        edge_results.append({"label": ec["label"], "result": r})

    _save_md(y_true_arr, y_scores_arr, y_pred_arr, per_sample, auc, f1, cm, edge_results, errors)
    print(f"Results saved to {_RESULTS_MD}")


def _save_md(y_true, y_scores, y_pred, per_sample, auc, f1, cm, edge_results, errors):
    lines = [
        "# Phase 1.8 Validation Results -- Corrected (HC3 Real Data)",
        "",
        "> Note: Previous synthetic self-written AI test numbers have been completely replaced.",
        "",
        "## Test Set Composition",
        "- Source: Hello-SimpleAI/HC3 (held out from calibration)",
        f"- Human samples: {int((y_true == 0).sum())}",
        f"- AI samples: {int((y_true == 1).sum())}",
        f"- Total scored: {len(y_true)}",
        "",
        "## Performance Metrics (threshold = 50)",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| ROC-AUC | {auc:.4f} |",
        f"| F1-score (threshold=50) | {f1:.4f} |",
        "",
        "## Confusion Matrix (threshold = 50)",
        "",
        "| | Predicted Human | Predicted AI |",
        "|---|---|---|",
        f"| **Actually Human** | TN={cm[0,0]} | FP={cm[0,1]} |",
        f"| **Actually AI** | FN={cm[1,0]} | TP={cm[1,1]} |",
        "",
        "## Per-Sample Scores",
        "",
        "| # | True Label | Score | Prediction | Correct |",
        "|---|---|---|---|---|",
    ]
    for i, ps in enumerate(per_sample):
        correct = "YES" if ps["correct"] else "NO"
        lines.append(f"| {i+1} | {ps['label']} | {ps['score']:.1f} | {ps['pred']} | {correct} |")

    lines += ["", "## Edge Case Results", ""]
    for er in edge_results:
        lines.append(f"### {er['label']}")
        r = er["result"]
        lines += [
            f"- text_score: {r['text_score']:.1f}",
            f"- curvature_raw: {r['signals']['curvature_raw']}",
            f"- burstiness_raw: {r['signals']['burstiness_raw']}",
            f"- cliche_density_pct: {r['signals']['cliche_density_pct']:.2f}%",
            f"- entropy: {r['signals']['entropy']:.3f} bits",
            "",
        ]

    _RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_hc3_validation()
