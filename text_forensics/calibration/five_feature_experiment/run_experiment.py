"""
text_forensics/calibration/five_feature_experiment/run_experiment.py

Five-Feature Ablation Experiment: Exclude ngram_repetition.

Uses IDENTICAL dataset, split, seed, scaler, and LR architecture as the
six-feature model — the only change is ngram_repetition is dropped from X.

Saves all artifacts to five_feature_experiment/ (separate from six_feature_model/).
The existing six-feature model is NEVER read from or written to.
"""

from __future__ import annotations

import json
import logging
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE    = Path(__file__).resolve().parent          # five_feature_experiment/
_CALIB   = _HERE.parent                             # calibration/
_PROJ    = _CALIB.parent.parent                     # project root
sys.path.insert(0, str(_PROJ))

# ---------------------------------------------------------------------------
# Canonical 6-feature order (same as six-feature model)  →  drop index 3
# ---------------------------------------------------------------------------
SIX_FEATURE_ORDER = [
    "curvature",
    "burstiness",
    "lexical_entropy",
    "ngram_repetition",       # ← index 3, excluded here
    "structural_regularity",
    "cliche_density",
]
DROP_FEATURE  = "ngram_repetition"
DROP_IDX      = SIX_FEATURE_ORDER.index(DROP_FEATURE)

FIVE_FEATURE_ORDER = [f for f in SIX_FEATURE_ORDER if f != DROP_FEATURE]

# Hyperparameters — IDENTICAL to six-feature model
RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
MAX_SAMPLES_PER_CLASS = None   # use full corpus, same as six-feature model

# Problematic paragraph
PROBLEMATIC_TEXT = (
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
# Re-use the six-feature cache to avoid re-extracting features
# ---------------------------------------------------------------------------
_SIX_CACHE = _CALIB / "six_feature_model" / "feature_cache.json"


def _load_cache() -> dict:
    if _SIX_CACHE.exists():
        with open(_SIX_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    logger.warning("six_feature_model/feature_cache.json not found — will re-extract (slow).")
    return {}


# ---------------------------------------------------------------------------
# Data loading  (identical logic to train_six_feature_model.py)
# ---------------------------------------------------------------------------

def _load_corpus(path: Path, label: int) -> list[tuple[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                t = item.get("text") or item.get("answer") or item.get("content") or ""
                if t:
                    texts.append(t)
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, str):
                texts.append(v)
            elif isinstance(v, list):
                texts.extend([x for x in v if isinstance(x, str)])
    texts = [t for t in texts if len(t.split()) >= 30]
    return [(t, label) for t in texts]


def _split_list(lst: list) -> tuple[list, list, list]:
    n = len(lst)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]


# ---------------------------------------------------------------------------
# Feature extraction — uses six-feature cache, drops ngram_repetition column
# ---------------------------------------------------------------------------

def _extract_row_from_cache(text: str, cache: dict) -> list | None:
    key = str(hash(text[:200]))
    row = cache.get(key)
    if row is None or not isinstance(row, list) or len(row) != 6:
        return None
    return row


def _extract_row_fresh(text: str) -> list | None:
    """Full extraction for texts not in cache."""
    try:
        from text_forensics.signals.curvature import get_curvature
        from text_forensics.signals.burstiness import get_burstiness
        from text_forensics.signals.lexical_entropy import get_lexical_stats
        from text_forensics.signals.ngram_repetition import get_ngram_repetition
        from text_forensics.signals.structural_regularity import get_structural_regularity
        from text_forensics.signals.cliche_scanner import get_cliche_density

        curv   = get_curvature(text)
        burst  = get_burstiness(text)
        ent    = get_lexical_stats(text)["entropy"]
        ngram  = get_ngram_repetition(text)["composite"]
        struct = get_structural_regularity(text)["composite"]
        cliche = get_cliche_density(text)
        return [curv, burst, ent, ngram, struct, cliche]
    except Exception as e:
        logger.warning("Extraction error: %s", e)
        return None


def extract_five_features(
    samples: list[tuple[str, int]],
    cache: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 5-feature vectors. Reads 6-feature cache, drops ngram_repetition column.
    Falls back to fresh extraction for cache misses.
    """
    rows, labels = [], []
    cache_hits, cache_misses = 0, 0

    for i, (text, label) in enumerate(samples):
        if (i + 1) % 50 == 0:
            logger.info("  [%d/%d] extracting... (hits=%d misses=%d)",
                        i+1, len(samples), cache_hits, cache_misses)

        row6 = _extract_row_from_cache(text, cache)
        if row6 is not None:
            cache_hits += 1
        else:
            row6 = _extract_row_fresh(text)
            cache_misses += 1

        if row6 is not None:
            # Drop ngram_repetition at index 3
            row5 = [v for i, v in enumerate(row6) if i != DROP_IDX]
            rows.append(row5)
            labels.append(label)

    logger.info("Extraction done: %d rows | cache_hits=%d misses=%d",
                len(rows), cache_hits, cache_misses)
    return np.array(rows, dtype=float), np.array(labels, dtype=int)


def _impute(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    col_means = np.nanmean(X_train, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    X_imp = X_train.copy()
    for j in range(X_imp.shape[1]):
        X_imp[np.isnan(X_imp[:, j]), j] = col_means[j]
    return X_imp, col_means


def _apply_means(X: np.ndarray, col_means: np.ndarray) -> np.ndarray:
    X_out = X.copy()
    for j in range(X_out.shape[1]):
        X_out[np.isnan(X_out[:, j]), j] = col_means[j]
    return X_out


# ---------------------------------------------------------------------------
# Threshold calibration  (identical logic to six-feature training script)
# ---------------------------------------------------------------------------

def _calibrate(model, scaler, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    X_sc = scaler.transform(X_val)
    proba = model.predict_proba(X_sc)[:, 1]

    best_t_ai, best_t_h, best_score = 0.70, 0.30, -1.0
    for t_h in np.arange(0.20, 0.55, 0.05):
        for t_ai in np.arange(0.55, 0.85, 0.05):
            if t_ai <= t_h:
                continue
            pred = np.where(proba >= t_ai, 1, np.where(proba <= t_h, 0, -1))
            mask = pred >= 0
            if mask.sum() < 10:
                continue
            f1  = f1_score(y_val[mask], pred[mask], average="binary", zero_division=0)
            fpr = ((pred == 1) & (y_val == 0)).sum() / max((y_val == 0).sum(), 1)
            sc  = f1 - 0.5 * fpr
            if sc > best_score:
                best_score = sc
                best_t_ai  = float(round(t_ai, 2))
                best_t_h   = float(round(t_h, 2))

    mid = round((best_t_h + best_t_ai) / 2.0, 2)
    logger.info("Calibrated: human_max=%.2f  likely_boundary=%.2f  ai_min=%.2f",
                best_t_h, mid, best_t_ai)
    return {"human_max": best_t_h, "likely_human_max": mid,
            "likely_ai_min": mid, "ai_min": best_t_ai}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y_true: np.ndarray, proba: np.ndarray, thresholds: dict, label: str) -> dict:
    t_ai = thresholds["ai_min"]
    t_h  = thresholds["human_max"]
    pred = np.where(proba >= t_ai, 1, np.where(proba <= t_h, 0, -1))
    mask = pred >= 0
    n_def = int(mask.sum())

    if n_def == 0:
        return {"label": label, "error": "no definite predictions"}

    y_d, p_d = y_true[mask], pred[mask]
    try:
        roc = roc_auc_score(y_true, proba)
    except Exception:
        roc = float("nan")

    acc   = accuracy_score(y_d, p_d)
    prec  = precision_score(y_d, p_d, zero_division=0)
    rec   = recall_score(y_d, p_d, zero_division=0)
    f1    = f1_score(y_d, p_d, zero_division=0)

    cm    = confusion_matrix(y_d, p_d, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr   = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr   = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "label":      label,
        "n_total":    int(len(y_true)),
        "n_definite": n_def,
        "accuracy":   round(float(acc),  4),
        "precision":  round(float(prec), 4),
        "recall":     round(float(rec),  4),
        "f1":         round(float(f1),   4),
        "roc_auc":    round(float(roc),  4),
        "fpr":        round(float(fpr),  4),
        "fnr":        round(float(fnr),  4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(m5: dict, m6: dict, thresholds: dict, coefs: dict,
                  prob_result: dict, meta: dict) -> str:
    lines = []
    lines.append("# Five-Feature Ablation Experiment (Without ngram_repetition)")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nDataset:   {meta['dataset_description']}")
    lines.append(f"Split:     {meta['train_n']} train / {meta['val_n']} val / {meta['test_n']} test")
    lines.append(f"Features:  {', '.join(FIVE_FEATURE_ORDER)}")
    lines.append(f"Excluded:  {DROP_FEATURE}")
    lines.append("")

    def row(m):
        return (f"| {m.get('label','?'):<40} | {m.get('accuracy','-')!s:>8} | "
                f"{m.get('precision','-')!s:>9} | {m.get('recall','-')!s:>6} | "
                f"{m.get('f1','-')!s:>6} | {m.get('roc_auc','-')!s:>7} | "
                f"{m.get('fpr','-')!s:>5} | {m.get('fnr','-')!s:>5} |")

    header = "| Model" + " " * 35 + "| Accuracy | Precision | Recall |   F1  | ROC-AUC |  FPR |  FNR |"
    sep    = "|" + "-"*41 + "|" + "-"*10 + "|" + "-"*11 + "|" + "-"*8 + "|" + "-"*7 + "|" + "-"*9 + "|" + "-"*6 + "|" + "-"*6 + "|"

    lines.append("## Comparison: Five-Feature vs Six-Feature (Test Set)\n")
    lines.append(header)
    lines.append(sep)
    lines.append(row(m5))
    lines.append(row(m6))
    lines.append("")

    # Delta row
    def delta(key):
        v5 = m5.get(key)
        v6 = m6.get(key)
        if v5 is None or v6 is None: return "N/A"
        d = round(v5 - v6, 4)
        return f"{d:+.4f}"

    lines.append("### Delta (Five-Feature minus Six-Feature)\n")
    lines.append("| Metric   | Delta |")
    lines.append("|----------|-------|")
    for k in ["accuracy","precision","recall","f1","roc_auc","fpr","fnr"]:
        lines.append(f"| {k:<9} | {delta(k)} |")
    lines.append("")

    # Confusion matrix
    cm5 = m5.get("confusion_matrix", {})
    lines.append("## Confusion Matrix (Five-Feature, Test Set)\n")
    lines.append("```")
    lines.append(f"                 Predicted Human  Predicted AI")
    lines.append(f"Actual Human         {cm5.get('tn',0):>5}          {cm5.get('fp',0):>5}")
    lines.append(f"Actual AI            {cm5.get('fn',0):>5}          {cm5.get('tp',0):>5}")
    lines.append("```")
    lines.append("")
    lines.append(f"n_definite = {m5.get('n_definite','?')} / {m5.get('n_total','?')} total  "
                 f"(remaining {m5.get('n_total',0)-m5.get('n_definite',0)} in uncertain zone)")
    lines.append("")

    lines.append("## Calibrated Thresholds (Five-Feature LR)\n")
    lines.append(f"- Human:        P(AI) ≤ {thresholds['human_max']:.2f}")
    lines.append(f"- Likely Human: {thresholds['human_max']:.2f} < P(AI) < {thresholds['likely_ai_min']:.2f}")
    lines.append(f"- Likely AI:    {thresholds['likely_ai_min']:.2f} ≤ P(AI) < {thresholds['ai_min']:.2f}")
    lines.append(f"- AI:           P(AI) ≥ {thresholds['ai_min']:.2f}")
    lines.append("")

    lines.append("## Learned Coefficients (Five-Feature LR)\n")
    lines.append("| Feature | Coefficient | Direction |")
    lines.append("|---------|-------------|-----------|\n")
    for fname, c in coefs.items():
        dir_ = "→ AI-like (higher)" if c > 0 else "→ Human-like (higher)"
        lines.append(f"| {fname} | {c:+.4f} | {dir_} |")
    lines.append("")
    lines.append("> Note: Signs after StandardScaler standardization. "
                 "Positive = feature above its mean increases P(AI).")
    lines.append("")

    lines.append("## Problematic AI Technical Paragraph — Regression Test\n")
    pr = prob_result
    lines.append(f"**P(AI) — Six-Feature model:**  ~1.0000 (AI)")
    lines.append(f"**P(AI) — Five-Feature model:**  {pr.get('ai_prob', 'N/A')}")
    lines.append(f"**Verdict — Five-Feature:**      {pr.get('verdict', 'N/A')}")
    lines.append("")
    lines.append("Feature values on problematic paragraph (5 features):")
    for fname in FIVE_FEATURE_ORDER:
        val = pr.get("features", {}).get(fname, "N/A")
        lines.append(f"  - {fname}: {val}")
    lines.append("")

    lines.append("## Notes\n")
    lines.append("- The six-feature model files (model.pkl, scaler.pkl, metadata) were not read or modified.")
    lines.append("- This experiment uses the same random seed, split ratios, and feature cache as the six-feature model.")
    lines.append("- All differences in results are attributable solely to the exclusion of ngram_repetition.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    _HERE.mkdir(exist_ok=True)
    logger.info("="*60)
    logger.info("FIVE-FEATURE ABLATION EXPERIMENT (without ngram_repetition)")
    logger.info("="*60)

    # Load cache (from six-feature run)
    cache = _load_cache()
    logger.info("Cache loaded: %d entries", len(cache))

    # Load corpora (same as six-feature model)
    random.seed(RANDOM_SEED)
    genre_h = _load_corpus(_CALIB / "genre_corpus_additions_human.json", 0)
    genre_a = _load_corpus(_CALIB / "genre_corpus_additions_ai.json",    1)
    hc3_h   = _load_corpus(_CALIB / "hc3_test_pool_human.json",          0)
    hc3_a   = _load_corpus(_CALIB / "hc3_test_pool_ai.json",             1)

    human_samples = genre_h + hc3_h
    ai_samples    = genre_a + hc3_a
    logger.info("Corpus: %d human, %d AI", len(human_samples), len(ai_samples))

    # Identical shuffle + split
    random.shuffle(human_samples)
    random.shuffle(ai_samples)

    h_train, h_val, h_test = _split_list(human_samples)
    a_train, a_val, a_test = _split_list(ai_samples)

    train_samples = h_train + a_train
    val_samples   = h_val   + a_val
    test_samples  = h_test  + a_test
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    logger.info("Split: train=%d  val=%d  test=%d",
                len(train_samples), len(val_samples), len(test_samples))

    # Extract 5-feature vectors
    logger.info("--- TRAINING SET ---")
    X_tr_raw, y_tr = extract_five_features(train_samples, cache)
    logger.info("--- VALIDATION SET ---")
    X_va_raw, y_va = extract_five_features(val_samples,   cache)
    logger.info("--- TEST SET ---")
    X_te_raw, y_te = extract_five_features(test_samples,  cache)

    # Impute NaN with training column means
    X_tr, col_means = _impute(X_tr_raw)
    X_va = _apply_means(X_va_raw, col_means)
    X_te = _apply_means(X_te_raw, col_means)

    # Fit scaler on training only
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    # Train LR (identical hyperparams)
    model = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        random_state=RANDOM_SEED, solver="lbfgs",
    )
    model.fit(X_tr_sc, y_tr)
    logger.info("Training complete.")

    # Calibrate thresholds on validation set
    thresholds = _calibrate(model, scaler, X_va, y_va)

    # Evaluate on test set
    proba_te = model.predict_proba(X_te_sc)[:, 1]
    m5 = _metrics(y_te, proba_te, thresholds, "Five-Feature LR (without ngram_repetition)")

    # Six-feature reference metrics (from stored metadata — NOT loading the model)
    six_meta_path = _CALIB / "six_feature_model" / "model_metadata.json"
    if six_meta_path.exists():
        with open(six_meta_path, "r", encoding="utf-8") as f:
            six_meta = json.load(f)
        m6 = six_meta["test_metrics"]
        m6["label"] = "Six-Feature LR (reference)"
    else:
        m6 = {"label": "Six-Feature LR (metadata not found)"}

    logger.info("Test metrics: %s", m5)

    # Coefficients
    coef = model.coef_[0]
    coefficients = {FIVE_FEATURE_ORDER[i]: round(float(coef[i]), 4) for i in range(5)}
    logger.info("Coefficients: %s", coefficients)

    # Problematic paragraph
    logger.info("Testing problematic AI technical paragraph...")
    try:
        row6 = _extract_row_fresh(PROBLEMATIC_TEXT)
        if row6 is not None:
            row5 = [v for i, v in enumerate(row6) if i != DROP_IDX]
            row5_imp = [v if v is not None else col_means[j] for j, v in enumerate(row5)]
            X_prob = np.array(row5_imp).reshape(1, -1)
            X_prob_sc = scaler.transform(X_prob)
            ai_prob = float(model.predict_proba(X_prob_sc)[0][1])
            t_ai = thresholds["ai_min"]
            t_h  = thresholds["human_max"]
            if ai_prob >= t_ai:
                verdict = "AI"
            elif ai_prob >= thresholds["likely_ai_min"]:
                verdict = "Likely AI"
            elif ai_prob <= t_h:
                verdict = "Human"
            else:
                verdict = "Likely Human"
            prob_result = {
                "ai_prob":   round(ai_prob, 4),
                "verdict":   verdict,
                "features":  {FIVE_FEATURE_ORDER[j]: round(v, 4) if v is not None else None
                              for j, v in enumerate(row5)},
            }
        else:
            prob_result = {"error": "extraction failed"}
    except Exception as e:
        prob_result = {"error": str(e)}
    logger.info("Problematic paragraph: %s", prob_result)

    # Save artifacts
    meta = {
        "experiment":       "Five-Feature Ablation (without ngram_repetition)",
        "feature_order":    FIVE_FEATURE_ORDER,
        "excluded_feature": DROP_FEATURE,
        "feature_means":    {FIVE_FEATURE_ORDER[i]: float(col_means[i]) for i in range(5)},
        "coefficients":     coefficients,
        "intercept":        float(model.intercept_[0]),
        "thresholds_4way":  thresholds,
        "train_n":          len(y_tr),
        "val_n":            len(y_va),
        "test_n":           len(y_te),
        "random_seed":      RANDOM_SEED,
        "model_class":      "LogisticRegression",
        "model_params":     {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
        "dataset_description": (
            "IDENTICAL to six-feature model: genre_corpus_additions_human/ai.json + "
            "hc3_test_pool_human/ai.json. Same seed, same split."
        ),
        "test_metrics":      m5,
        "six_feature_test_metrics": m6,
        "problematic_paragraph_result": prob_result,
        "trained_date":     datetime.now().strftime("%Y-%m-%d"),
        "note": "Experimental only. Production six-feature model is unchanged.",
    }

    with open(_HERE / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(_HERE / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(_HERE / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report = _write_report(m5, m6, thresholds, coefficients, prob_result, meta)
    with open(_HERE / "evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    # Print final summary
    logger.info("")
    logger.info("="*60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*60)
    logger.info("Five-Feature Test Results:")
    logger.info("  Accuracy  : %.4f", m5.get("accuracy",0))
    logger.info("  Precision : %.4f", m5.get("precision",0))
    logger.info("  Recall    : %.4f", m5.get("recall",0))
    logger.info("  F1        : %.4f", m5.get("f1",0))
    logger.info("  ROC-AUC   : %.4f", m5.get("roc_auc",0))
    logger.info("  FPR       : %.4f", m5.get("fpr",0))
    logger.info("  FNR       : %.4f", m5.get("fnr",0))
    logger.info("")
    logger.info("Six-Feature Reference:")
    logger.info("  F1        : %.4f  ROC-AUC : %.4f",
                m6.get("f1",0), m6.get("roc_auc",0))
    logger.info("")
    logger.info("Problematic paragraph: P(AI)=%.4f  Verdict=%s",
                prob_result.get("ai_prob", 0), prob_result.get("verdict", "N/A"))
    logger.info("")
    logger.info("Artifacts saved to: %s", _HERE)


if __name__ == "__main__":
    run()
