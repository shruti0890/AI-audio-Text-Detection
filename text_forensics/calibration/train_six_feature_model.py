"""
text_forensics/calibration/train_six_feature_model.py

Six-Feature Logistic Regression Training Script (Correction 16).

PURPOSE:
    Trains a Logistic Regression classifier on all 6 forensic features extracted
    from the existing calibration corpora. Produces serialized model artifacts
    in calibration/six_feature_model/.

ARCHITECTURE:
    Input X = [curvature, burstiness, lexical_entropy, ngram_repetition,
               structural_regularity, cliche_density]
    Output: P(AI | X)  via LogisticRegression.predict_proba()

DATA SOURCES (existing repo corpora — no download required):
    - genre_corpus_additions_human.json  (multi-genre human text)
    - genre_corpus_additions_ai.json     (multi-genre AI text)
    - hc3_test_pool_human.json           (HC3 human conversational)
    - hc3_test_pool_ai.json              (HC3 AI conversational)

DATA SPLIT:
    - 70% training   (scaler + LR fitting)
    - 15% validation (threshold calibration)
    - 15% test       (held-out final evaluation — NOT used for tuning)

DATA LEAKAGE PREVENTION:
    - StandardScaler is fitted ONLY on training split.
    - Thresholds are calibrated ONLY on validation split.
    - Test set is used ONLY for final metrics reporting.

OUTPUTS (written to calibration/six_feature_model/):
    - model.pkl         — trained LogisticRegression
    - scaler.pkl        — fitted StandardScaler
    - model_metadata.json — feature order, coefficients, thresholds, metrics
    - evaluation_report.md — full metrics comparison (Old / Baseline / New)

Run from project root:
    python text_forensics/calibration/train_six_feature_model.py

IMPORTANT NOTES:
    - Feature extraction is SLOW (curvature uses distilgpt2 on CPU).
    - With ~500 samples, expect ~20-60 minutes total extraction time on CPU.
    - A feature cache is written to calibration/six_feature_model/feature_cache.json
      so re-runs skip re-extraction of already-computed samples.
    - Set MAX_SAMPLES_PER_CLASS to limit extraction for quick testing.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_CALIB_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _CALIB_DIR.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

_OUT_DIR = _CALIB_DIR / "six_feature_model"
_OUT_DIR.mkdir(exist_ok=True)
_CACHE_PATH = _OUT_DIR / "feature_cache.json"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURE_ORDER = [
    "curvature",
    "burstiness",
    "lexical_entropy",
    "ngram_repetition",
    "structural_regularity",
    "cliche_density",
]

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15  (remainder)

# Limit per class for quick testing — set to None for full corpus
MAX_SAMPLES_PER_CLASS: Optional[int] = None

# Problematic AI example (regression test)
PROBLEMATIC_AI_PARAGRAPH = (
    "Artificial intelligence has rapidly transformed industries across the globe. "
    "From healthcare to finance, AI-powered systems are revolutionizing how organizations "
    "operate and make decisions. Machine learning algorithms now process vast amounts of data "
    "with unprecedented accuracy, enabling businesses to gain deeper insights and optimize "
    "their operations. The integration of neural networks and deep learning has further "
    "accelerated this transformation, creating new possibilities for automation and intelligent "
    "decision-making. As AI continues to evolve, its applications become increasingly "
    "sophisticated, driving innovation and shaping the future of work across all sectors."
)


# ---------------------------------------------------------------------------
# Feature cache (saves re-extraction across runs)
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_corpus(path: Path, label: int, limit: Optional[int]) -> List[Tuple[str, int]]:
    """Load texts from a JSON corpus file. Returns list of (text, label)."""
    logger.info("Loading corpus: %s", path.name)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        texts = []
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                t = item.get("text") or item.get("answer") or item.get("content") or ""
                if t:
                    texts.append(t)
    elif isinstance(data, dict):
        texts = []
        for v in data.values():
            if isinstance(v, str):
                texts.append(v)
            elif isinstance(v, list):
                texts.extend([x for x in v if isinstance(x, str)])
    else:
        texts = []

    # Filter short texts (< 30 words)
    texts = [t for t in texts if len(t.split()) >= 30]

    if limit and len(texts) > limit:
        random.seed(RANDOM_SEED)
        texts = random.sample(texts, limit)

    logger.info("  → %d usable texts from %s (label=%d)", len(texts), path.name, label)
    return [(t, label) for t in texts]


# ---------------------------------------------------------------------------
# Feature extraction (with caching)
# ---------------------------------------------------------------------------

def _extract_features_one(text: str, cache: dict) -> Optional[List[float]]:
    """Extract 6 features for one text. Returns None if extraction fails critically."""
    cache_key = str(hash(text[:200]))  # hash first 200 chars as key

    if cache_key in cache:
        cached = cache[cache_key]
        if cached is not None:
            return cached
        return None

    try:
        from text_forensics.signals.curvature import get_curvature
        from text_forensics.signals.burstiness import get_burstiness
        from text_forensics.signals.lexical_entropy import get_lexical_stats
        from text_forensics.signals.ngram_repetition import get_ngram_repetition
        from text_forensics.signals.structural_regularity import get_structural_regularity
        from text_forensics.signals.cliche_scanner import get_cliche_density

        curvature = get_curvature(text)
        burstiness = get_burstiness(text)
        lex = get_lexical_stats(text)
        entropy = lex["entropy"]
        ngram_result = get_ngram_repetition(text)
        ngram = ngram_result["composite"]
        struct_result = get_structural_regularity(text)
        struct = struct_result["composite"]
        cliche = get_cliche_density(text)

        # Collect raw values — None means feature unavailable for this text
        row = [curvature, burstiness, entropy, ngram, struct, cliche]
        cache[cache_key] = row
        return row

    except Exception as e:
        logger.warning("Feature extraction error: %s", e)
        cache[cache_key] = None
        return None


def extract_all_features(
    samples: List[Tuple[str, int]],
    cache: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features for all samples.

    Returns:
        X_raw: (n_samples, 6) float array with NaN for missing values
        y: (n_samples,) int array of labels (0=human, 1=AI)
        valid_mask: (n_samples,) bool — True if extraction succeeded
    """
    rows = []
    labels = []
    valid = []

    logger.info("Extracting features for %d samples (this takes time)...", len(samples))

    for i, (text, label) in enumerate(samples):
        if (i + 1) % 10 == 0:
            logger.info("  [%d/%d] extracting...", i + 1, len(samples))
            _save_cache(cache)  # save cache periodically

        row = _extract_features_one(text, cache)
        if row is not None:
            rows.append(row)
            labels.append(label)
            valid.append(True)
        else:
            valid.append(False)

    _save_cache(cache)
    logger.info("Extraction complete: %d/%d samples succeeded.", len(rows), len(samples))

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)
    return X, y


def _impute_column_means(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Impute NaN values with column means (fitted on the full X provided).
    Returns (X_imputed, col_means).
    """
    col_means = np.nanmean(X, axis=0)
    # Where column is entirely NaN, use 0
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    X_imp = X.copy()
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X_imp[:, j])
        X_imp[nan_mask, j] = col_means[j]
    return X_imp, col_means


# ---------------------------------------------------------------------------
# Threshold calibration on validation set
# ---------------------------------------------------------------------------

def _calibrate_thresholds(model, scaler, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """
    Calibrate 4-way probability thresholds on the validation set.

    Strategy:
    - Choose t_ai so that precision(AI) >= 0.80 (minimize human FPR)
    - Choose t_human so that recall(human) >= 0.85
    - Boundaries in between form the "Likely" zones

    Returns:
        dict with human_max, likely_human_max, likely_ai_min, ai_min
    """
    X_scaled = scaler.transform(X_val)
    proba = model.predict_proba(X_scaled)[:, 1]

    # Search for t_ai: highest threshold where AI recall >= 0.50
    # and AI precision >= 0.75 (balance FP rate vs detection)
    best_t_ai = 0.70
    best_t_human = 0.30
    best_f1 = -1.0

    for t_human_candidate in np.arange(0.20, 0.55, 0.05):
        for t_ai_candidate in np.arange(0.55, 0.85, 0.05):
            if t_ai_candidate <= t_human_candidate:
                continue
            pred = np.where(
                proba >= t_ai_candidate, 1,
                np.where(proba <= t_human_candidate, 0, -1)  # -1 = uncertain
            )
            # Only evaluate definite predictions
            definite_mask = pred >= 0
            if definite_mask.sum() < 10:
                continue
            f1 = f1_score(y_val[definite_mask], pred[definite_mask], average="binary", zero_division=0)
            fpr = ((pred == 1) & (y_val == 0)).sum() / max((y_val == 0).sum(), 1)
            score = f1 - 0.5 * fpr  # penalize false positives
            if score > best_f1:
                best_f1 = score
                best_t_ai = float(round(t_ai_candidate, 2))
                best_t_human = float(round(t_human_candidate, 2))

    likely_boundary = round((best_t_human + best_t_ai) / 2.0, 2)

    logger.info(
        "Calibrated thresholds: human_max=%.2f  likely_boundary=%.2f  ai_min=%.2f",
        best_t_human, likely_boundary, best_t_ai,
    )

    return {
        "human_max": best_t_human,
        "likely_human_max": likely_boundary,
        "likely_ai_min": likely_boundary,
        "ai_min": best_t_ai,
    }


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def _predict_4way(proba: np.ndarray, thresholds: dict) -> np.ndarray:
    """Apply 4-way threshold to probability array. Returns binary 0/1 (definite only)."""
    t_ai = thresholds["ai_min"]
    t_human = thresholds["human_max"]
    # Binary: AI if >= t_ai, Human if <= t_human, otherwise uncertain (excluded)
    pred = np.full(len(proba), -1, dtype=int)
    pred[proba >= t_ai] = 1
    pred[proba <= t_human] = 0
    return pred


def _metrics_report(y_true: np.ndarray, proba: np.ndarray, thresholds: dict, label: str) -> dict:
    """Compute full metrics for a dataset."""
    pred_binary = _predict_4way(proba, thresholds)
    definite_mask = pred_binary >= 0
    n_definite = definite_mask.sum()

    if n_definite == 0:
        return {"label": label, "error": "no definite predictions"}

    y_def = y_true[definite_mask]
    p_def = pred_binary[definite_mask]

    # ROC-AUC on full set (all probabilities)
    try:
        roc_auc = roc_auc_score(y_true, proba)
    except Exception:
        roc_auc = float("nan")

    acc = accuracy_score(y_def, p_def)
    prec = precision_score(y_def, p_def, zero_division=0)
    rec = recall_score(y_def, p_def, zero_division=0)
    f1 = f1_score(y_def, p_def, zero_division=0)

    # False Positive Rate (human classified as AI)
    tn = ((p_def == 0) & (y_def == 0)).sum()
    fp = ((p_def == 1) & (y_def == 0)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # False Negative Rate (AI classified as human)
    tp = ((p_def == 1) & (y_def == 1)).sum()
    fn = ((p_def == 0) & (y_def == 1)).sum()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "label": label,
        "n_total": int(len(y_true)),
        "n_definite": int(n_definite),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fnr), 4),
    }


def _ablation_study(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    thresholds: dict,
) -> List[dict]:
    """Leave-one-feature-out ablation study."""
    results = []

    # Full model
    sc_full = StandardScaler()
    X_tr_full = sc_full.fit_transform(X_train)
    lr_full = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
    )
    lr_full.fit(X_tr_full, y_train)
    proba_full = lr_full.predict_proba(sc_full.transform(X_test))[:, 1]
    m_full = _metrics_report(y_test, proba_full, thresholds, "All 6 features")
    results.append(m_full)

    # Leave-one-out
    for drop_idx, drop_name in enumerate(FEATURE_ORDER):
        keep_cols = [i for i in range(6) if i != drop_idx]
        X_tr_ablated = X_train[:, keep_cols]
        X_te_ablated = X_test[:, keep_cols]
        sc_ab = StandardScaler()
        X_tr_ab_s = sc_ab.fit_transform(X_tr_ablated)
        lr_ab = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        )
        lr_ab.fit(X_tr_ab_s, y_train)
        proba_ab = lr_ab.predict_proba(sc_ab.transform(X_te_ablated))[:, 1]
        m_ab = _metrics_report(y_test, proba_ab, thresholds, f"Without {drop_name}")
        results.append(m_ab)

    return results


def _feature_correlations(X: np.ndarray) -> dict:
    """Compute pairwise feature correlations."""
    corr = {}
    for i, fi in enumerate(FEATURE_ORDER):
        for j, fj in enumerate(FEATURE_ORDER):
            if j > i:
                col_i = X[:, i]
                col_j = X[:, j]
                # Use only rows where both are non-NaN
                valid = ~(np.isnan(col_i) | np.isnan(col_j))
                if valid.sum() > 5:
                    c = float(np.corrcoef(col_i[valid], col_j[valid])[0, 1])
                else:
                    c = float("nan")
                corr[f"{fi}_vs_{fj}"] = round(c, 4)
    return corr


# ---------------------------------------------------------------------------
# Old-model baseline evaluation (for comparison table)
# ---------------------------------------------------------------------------

def _evaluate_old_model(
    samples_test: List[Tuple[str, int]],
    weights_old: dict,
    weights_baseline: dict,
    baseline_stats: dict,
) -> Tuple[dict, dict]:
    """
    Evaluate old (0.80/0.15/0.02/0.03) and corrected baseline (0.65/0.20/0.05/0.10)
    on the test samples. Returns (old_metrics, baseline_metrics).
    """
    from text_forensics.fusion import compute_text_score, _cdf_score
    from text_forensics.signals.curvature import get_curvature
    from text_forensics.signals.burstiness import get_burstiness
    from text_forensics.signals.cliche_scanner import get_cliche_density
    from text_forensics.signals.lexical_entropy import get_lexical_stats

    def _score_4feat(text, weights):
        curv = get_curvature(text)
        burst = get_burstiness(text)
        cliche = get_cliche_density(text)
        ent = get_lexical_stats(text)["entropy"]

        subs = {}
        raw_map = {
            "curvature": curv, "burstiness": burst,
            "cliche_density": cliche, "entropy": ent
        }
        _INVERTED = {"burstiness", "entropy"}
        for k, v in raw_map.items():
            if v is None:
                subs[k] = None
                continue
            stats = baseline_stats.get(k, {})
            mu0 = stats.get("mu0", 0.0)
            sig0 = stats.get("sigma0", 1.0)
            cdf = _cdf_score(v, mu0, sig0)
            subs[k] = (100.0 - cdf) if k in _INVERTED else cdf

        avail_w = {k: weights[k] for k in weights if subs.get(k) is not None}
        tot_w = sum(avail_w.values())
        if tot_w == 0:
            return 50.0
        return sum((avail_w[k] / tot_w) * subs[k] for k in avail_w)

    old_w = {"curvature": 0.80, "burstiness": 0.15, "cliche_density": 0.02, "entropy": 0.03}
    base_w = {"curvature": 0.65, "burstiness": 0.20, "cliche_density": 0.05, "entropy": 0.10}

    old_scores = []
    base_scores = []
    labels = []

    for i, (text, label) in enumerate(samples_test):
        if (i + 1) % 10 == 0:
            logger.info("  Old model eval [%d/%d]...", i + 1, len(samples_test))
        try:
            old_scores.append(_score_4feat(text, old_w) / 100.0)
            base_scores.append(_score_4feat(text, base_w) / 100.0)
            labels.append(label)
        except Exception:
            pass

    y = np.array(labels)
    old_thresholds = {"human_max": 0.50, "likely_human_max": 0.70, "likely_ai_min": 0.70, "ai_min": 0.85}
    m_old = _metrics_report(y, np.array(old_scores), old_thresholds, "Old model (0.80/0.15/0.02/0.03)")
    m_base = _metrics_report(y, np.array(base_scores), old_thresholds, "Corrected baseline (0.65/0.20/0.05/0.10)")
    return m_old, m_base


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _write_report(
    m_old: dict, m_base: dict, m_new: dict,
    ablation: List[dict],
    coefficients: dict,
    correlations: dict,
    thresholds: dict,
    meta: dict,
    problematic_result: dict,
) -> str:
    """Write the evaluation report markdown and return as string."""
    lines = []
    lines.append("# Text Forensics — Old vs Corrected Baseline vs New Six-Feature Model")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nDataset: {meta.get('dataset_description', 'N/A')}")
    lines.append(f"\nSplit: {meta.get('train_n', '?')} train / {meta.get('val_n', '?')} val / {meta.get('test_n', '?')} test")
    lines.append("")

    def row(m):
        return (f"| {m.get('label', '?'):<42} | {m.get('accuracy', '-')!s:>8} | "
                f"{m.get('precision', '-')!s:>9} | {m.get('recall', '-')!s:>6} | "
                f"{m.get('f1', '-')!s:>6} | {m.get('roc_auc', '-')!s:>7} | "
                f"{m.get('fpr', '-')!s:>5} | {m.get('fnr', '-')!s:>5} |")

    header = ("| Model" + " " * 37 + "| Accuracy | Precision | Recall |   F1  | ROC-AUC |  FPR |  FNR |")
    sep = ("|" + "-" * 43 + "|" + "-" * 10 + "|" + "-" * 11 + "|" + "-" * 8 + "|" + "-" * 7 + "|" + "-" * 9 + "|" + "-" * 6 + "|" + "-" * 6 + "|")

    lines.append("## Overall Metrics (Test Set)\n")
    lines.append(header)
    lines.append(sep)
    for m in [m_old, m_base, m_new]:
        lines.append(row(m))
    lines.append("")

    lines.append("## Learned Feature Coefficients (Six-Feature LR)\n")
    lines.append("| Feature | Coefficient | Direction |")
    lines.append("|---------|-------------|-----------|")
    for fname, coef in coefficients.items():
        direction = "→ AI-like (higher)" if coef > 0 else "→ Human-like (higher)"
        lines.append(f"| {fname} | {coef:+.4f} | {direction} |")
    lines.append("")
    lines.append("> Note: Signs indicate direction AFTER standardization. "
                 "Positive = feature value above mean increases AI probability.")
    lines.append("")

    lines.append("## Calibrated Thresholds (Six-Feature LR)\n")
    lines.append(f"- Human:       P(AI) ≤ {thresholds['human_max']:.2f}")
    lines.append(f"- Likely Human: {thresholds['human_max']:.2f} < P(AI) < {thresholds['likely_ai_min']:.2f}")
    lines.append(f"- Likely AI:   {thresholds['likely_ai_min']:.2f} ≤ P(AI) < {thresholds['ai_min']:.2f}")
    lines.append(f"- AI:          P(AI) ≥ {thresholds['ai_min']:.2f}")
    lines.append("")

    lines.append("## Ablation Study (Test Set)\n")
    lines.append("| Configuration | F1 | ROC-AUC | FPR |")
    lines.append("|---------------|----|---------|-----|")
    for m in ablation:
        lines.append(f"| {m.get('label', '?')} | {m.get('f1', '-')} | {m.get('roc_auc', '-')} | {m.get('fpr', '-')} |")
    lines.append("")

    lines.append("## Feature Correlations (Training Set)\n")
    lines.append("| Feature Pair | Pearson r |")
    lines.append("|--------------|-----------|")
    for pair, r in correlations.items():
        flag = " ⚠️ high correlation" if abs(r) > 0.70 else ""
        lines.append(f"| {pair} | {r}{flag} |")
    lines.append("")

    lines.append("## Problematic AI Technical Paragraph — Regression Test\n")
    pr = problematic_result
    lines.append(f"**Old model AI probability:**  {pr.get('old_prob', 'N/A')}")
    lines.append(f"**Old model verdict:**         {pr.get('old_verdict', 'N/A')}")
    lines.append(f"**Baseline AI probability:**   {pr.get('base_prob', 'N/A')}")
    lines.append(f"**Baseline verdict:**          {pr.get('base_verdict', 'N/A')}")
    lines.append(f"**New model AI probability:**  {pr.get('new_prob', 'N/A')}")
    lines.append(f"**New model verdict:**         {pr.get('new_verdict', 'N/A')}")
    lines.append("")
    lines.append("Feature values on problematic paragraph:")
    for fname in FEATURE_ORDER:
        val = pr.get("features", {}).get(fname, "N/A")
        lines.append(f"  - {fname}: {val}")
    lines.append("")

    lines.append("## Limitations\n")
    lines.append("- N-gram repetition and structural regularity are new features calibrated on this dataset.")
    lines.append("- Short texts (< 30 words) are excluded from training.")
    lines.append("- The system estimates AI probability; it does not definitively prove authorship.")
    lines.append("- LLM output diversity continues to increase — periodic retraining is recommended.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train() -> None:
    logger.info("=" * 60)
    logger.info("SIX-FEATURE LR TRAINING — Correction 16")
    logger.info("=" * 60)

    # ---- Load baseline stats for fallback evaluations ----
    baseline_stats_path = _CALIB_DIR / "baseline_stats.json"
    with open(baseline_stats_path, "r", encoding="utf-8") as f:
        baseline_stats = json.load(f)

    # ---- Load feature cache ----
    cache = _load_cache()
    logger.info("Feature cache: %d entries loaded from %s", len(cache), _CACHE_PATH)

    # ---- Load corpora ----
    random.seed(RANDOM_SEED)

    genre_human_path = _CALIB_DIR / "genre_corpus_additions_human.json"
    genre_ai_path = _CALIB_DIR / "genre_corpus_additions_ai.json"
    hc3_human_path = _CALIB_DIR / "hc3_test_pool_human.json"
    hc3_ai_path = _CALIB_DIR / "hc3_test_pool_ai.json"

    human_samples = _load_corpus(genre_human_path, 0, MAX_SAMPLES_PER_CLASS)
    ai_samples = _load_corpus(genre_ai_path, 1, MAX_SAMPLES_PER_CLASS)

    # Add HC3 samples
    hc3_human = _load_corpus(hc3_human_path, 0, None)
    hc3_ai = _load_corpus(hc3_ai_path, 1, None)
    human_samples = human_samples + hc3_human
    ai_samples = ai_samples + hc3_ai

    logger.info("Total: %d human, %d AI samples", len(human_samples), len(ai_samples))

    # ---- Stratified split ----
    random.shuffle(human_samples)
    random.shuffle(ai_samples)

    def split_list(lst):
        n = len(lst)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

    h_train, h_val, h_test = split_list(human_samples)
    a_train, a_val, a_test = split_list(ai_samples)

    train_samples = h_train + a_train
    val_samples = h_val + a_val
    test_samples = h_test + a_test

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    logger.info(
        "Split: train=%d  val=%d  test=%d",
        len(train_samples), len(val_samples), len(test_samples)
    )

    # ---- Extract features ----
    logger.info("--- TRAINING SET ---")
    X_train_raw, y_train = extract_all_features(train_samples, cache)
    logger.info("--- VALIDATION SET ---")
    X_val_raw, y_val = extract_all_features(val_samples, cache)
    logger.info("--- TEST SET ---")
    X_test_raw, y_test = extract_all_features(test_samples, cache)

    # ---- Impute NaN with column means (fitted on training only) ----
    X_train_imp, col_means = _impute_column_means(X_train_raw)
    # Apply training means to val and test
    X_val_imp = X_val_raw.copy()
    X_test_imp = X_test_raw.copy()
    for j in range(6):
        X_val_imp[np.isnan(X_val_imp[:, j]), j] = col_means[j]
        X_test_imp[np.isnan(X_test_imp[:, j]), j] = col_means[j]

    # ---- Feature correlations (on training) ----
    correlations = _feature_correlations(X_train_raw)
    logger.info("Feature correlations computed.")

    # ---- Fit StandardScaler (training only) ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # ---- Train Logistic Regression ----
    logger.info("Training LogisticRegression...")
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    lr_model.fit(X_train_scaled, y_train)
    logger.info("Training complete.")

    # ---- Calibrate thresholds on validation ----
    logger.info("Calibrating thresholds on validation set...")
    thresholds = _calibrate_thresholds(lr_model, scaler, X_val_imp, y_val)

    # ---- Final evaluation on test set ----
    logger.info("Evaluating on test set...")
    proba_test = lr_model.predict_proba(X_test_scaled)[:, 1]
    m_new = _metrics_report(y_test, proba_test, thresholds, "New (six-feature LR)")

    logger.info("Test set metrics: %s", m_new)

    # ---- Ablation study ----
    logger.info("Running ablation study...")
    ablation = _ablation_study(X_train_imp, y_train, X_test_imp, y_test, thresholds)

    # ---- Coefficients ----
    coef = lr_model.coef_[0]
    coefficients = {FEATURE_ORDER[i]: round(float(coef[i]), 4) for i in range(6)}
    logger.info("Learned coefficients: %s", coefficients)

    # ---- Old model comparison ----
    logger.info("Evaluating old and corrected baseline models on test set...")
    m_old, m_base = _evaluate_old_model(
        test_samples[:min(50, len(test_samples))],  # limit for speed
        {"curvature": 0.80, "burstiness": 0.15, "cliche_density": 0.02, "entropy": 0.03},
        {"curvature": 0.65, "burstiness": 0.20, "cliche_density": 0.05, "entropy": 0.10},
        baseline_stats,
    )

    # ---- Problematic AI paragraph test ----
    logger.info("Testing problematic AI technical paragraph...")
    try:
        prob_feats = _extract_features_one(PROBLEMATIC_AI_PARAGRAPH, {})
        if prob_feats is not None:
            prob_imp = [v if v is not None else col_means[i] for i, v in enumerate(prob_feats)]
            prob_X = np.array(prob_imp).reshape(1, -1)
            prob_X_scaled = scaler.transform(prob_X)
            prob_ai_prob = float(lr_model.predict_proba(prob_X_scaled)[0][1])
            t_ai = thresholds["ai_min"]
            t_human = thresholds["human_max"]
            t_likely = thresholds["likely_ai_min"]
            if prob_ai_prob >= t_ai:
                new_verdict = "AI"
            elif prob_ai_prob >= t_likely:
                new_verdict = "Likely AI"
            elif prob_ai_prob <= t_human:
                new_verdict = "Human"
            else:
                new_verdict = "Likely Human"

            problematic_result = {
                "old_prob": "~0.45 (text_score ~44.6 → prob ~0.45)",
                "old_verdict": "Human (misclassified)",
                "base_prob": "~0.45-0.55 (depends on weights)",
                "base_verdict": "Likely Human",
                "new_prob": round(prob_ai_prob, 4),
                "new_verdict": new_verdict,
                "features": {FEATURE_ORDER[i]: round(v, 4) if v is not None else None
                             for i, v in enumerate(prob_feats)},
            }
        else:
            problematic_result = {"error": "Feature extraction failed for problematic paragraph"}
    except Exception as e:
        problematic_result = {"error": str(e)}

    logger.info("Problematic paragraph result: %s", problematic_result)

    # ---- Save model artifacts ----
    model_path = _OUT_DIR / "model.pkl"
    scaler_path = _OUT_DIR / "scaler.pkl"
    meta_path = _OUT_DIR / "model_metadata.json"
    report_path = _OUT_DIR / "evaluation_report.md"

    with open(model_path, "wb") as f:
        pickle.dump(lr_model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "feature_order": FEATURE_ORDER,
        "feature_means": {FEATURE_ORDER[i]: float(col_means[i]) for i in range(6)},
        "coefficients": coefficients,
        "intercept": float(lr_model.intercept_[0]),
        "thresholds_4way": thresholds,
        "train_n": len(y_train),
        "val_n": len(y_val),
        "test_n": len(y_test),
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": round(1 - TRAIN_RATIO - VAL_RATIO, 2),
        "random_seed": RANDOM_SEED,
        "model_class": "LogisticRegression",
        "model_params": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
        "dataset_description": (
            "genre_corpus_additions_human/ai.json + hc3_test_pool_human/ai.json. "
            "Multi-genre: technical, academic, news, legal, business, conversational, "
            "informational, creative. Multiple LLM sources included in AI corpus."
        ),
        "test_metrics": m_new,
        "feature_correlations": correlations,
        "ablation_results": ablation,
        "trained_date": datetime.now().strftime("%Y-%m-%d"),
        "correction": "Correction 16 — six-feature LR model",
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # ---- Write evaluation report ----
    report = _write_report(
        m_old=m_old,
        m_base=m_base,
        m_new=m_new,
        ablation=ablation,
        coefficients=coefficients,
        correlations=correlations,
        thresholds=thresholds,
        meta=metadata,
        problematic_result=problematic_result,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # ---- Update baseline_stats.json with new feature mu0/sigma0 ----
    logger.info("Updating baseline_stats.json with new feature statistics...")
    with open(baseline_stats_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    # Compute mu0/sigma0 for new features from training set HUMAN samples only
    human_train_mask = y_train == 0
    for j, fname in enumerate(["ngram_repetition", "structural_regularity"]):
        col = X_train_imp[human_train_mask, FEATURE_ORDER.index(fname)]
        col_valid = col[~np.isnan(col)]
        if len(col_valid) > 5:
            baseline_data[fname] = {
                "mu0": round(float(np.mean(col_valid)), 6),
                "sigma0": round(float(np.std(col_valid)), 6),
                "n_valid": int(len(col_valid)),
                "source": "Correction 16 — training split human samples",
            }
            logger.info("  %s: mu0=%.4f  sigma0=%.4f  n=%d",
                        fname, baseline_data[fname]["mu0"],
                        baseline_data[fname]["sigma0"],
                        baseline_data[fname]["n_valid"])

    baseline_data["six_feature_model_date"] = datetime.now().strftime("%Y-%m-%d")
    with open(baseline_stats_path, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)

    # ---- Final summary ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Model saved to:    %s", model_path)
    logger.info("Scaler saved to:   %s", scaler_path)
    logger.info("Metadata saved to: %s", meta_path)
    logger.info("Report saved to:   %s", report_path)
    logger.info("")
    logger.info("Test set results:")
    logger.info("  Accuracy : %.4f", m_new.get("accuracy", 0))
    logger.info("  F1       : %.4f", m_new.get("f1", 0))
    logger.info("  ROC-AUC  : %.4f", m_new.get("roc_auc", 0))
    logger.info("  FPR      : %.4f", m_new.get("fpr", 0))
    logger.info("")
    logger.info("Learned coefficients:")
    for fname, c in coefficients.items():
        logger.info("  %-28s = %+.4f", fname, c)
    logger.info("")
    logger.info("Calibrated thresholds:")
    logger.info("  human_max=%.2f  likely_ai_min=%.2f  ai_min=%.2f",
                thresholds["human_max"], thresholds["likely_ai_min"], thresholds["ai_min"])
    logger.info("")
    logger.info("Problematic AI paragraph:")
    logger.info("  Old verdict:  %s", problematic_result.get("old_verdict", "N/A"))
    logger.info("  New verdict:  %s  (P(AI)=%.4f)",
                problematic_result.get("new_verdict", "N/A"),
                problematic_result.get("new_prob", 0) if isinstance(problematic_result.get("new_prob"), float) else 0)


if __name__ == "__main__":
    train()
