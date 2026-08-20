"""
text_forensics/calibration/train_modern_model.py

Parallel multi-process feature extraction and model training for the
Modern 5-Feature Logistic Regression model (using SmolLM2-135M for curvature).

Utilizes multi-processing across CPU cores with persistent incremental caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_CALIB_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _CALIB_DIR.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

_OUTPUT_DIR = _CALIB_DIR / "modern_model_experiment"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_PATH = _OUTPUT_DIR / "feature_cache.json"

FEATURE_ORDER = [
    "curvature",
    "burstiness",
    "lexical_entropy",
    "structural_regularity",
    "cliche_density",
]

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Benchmark test paragraphs
HUMAN_PAXOS_PARAGRAPH = (
    "In distributed systems, achieving consensus across asynchronous networks with unreliable "
    "nodes requires careful algorithm design. Paxos achieves this by having proposers send "
    "prepare requests with strictly increasing proposal numbers to a majority of acceptors. "
    "If an acceptor has not seen a higher proposal number, it promises not to accept future "
    "proposals with lower numbers and returns the highest-numbered value it has already accepted. "
    "Once a proposer collects promises from a quorum, it sends an accept request. Although the "
    "protocol guarantees safety even under packet loss or network partitions, liveness can be "
    "threatened by dueling proposers, which practical implementations typically resolve via "
    "leader election or randomized exponential backoff timeouts."
)

HUMAN_HISTORY_PARAGRAPH = (
    "The Treaty of Westphalia in 1648 ended the Thirty Years' War and established "
    "a new system of sovereign nation-states. Each state gained the right to choose its "
    "own religion without interference. The treaty's principles shaped European diplomacy "
    "for centuries. I remember reading about this in a dusty corner of the library, "
    "surrounded by old maps that smelled faintly of tobacco. The negotiations were "
    "extraordinarily complex — dozens of parties, each with competing interests. "
    "What struck me most was how chaotic the process was, nothing like the orderly "
    "narratives you read in textbooks. Delegates argued over precedence for months "
    "before any substantive talks began."
)

CHATGPT_TECHNICAL_PARAGRAPH = (
    "Augmenting the training corpus with multi-generator AI text substantially improved "
    "sensitivity to Claude-generated content and moderately improved sensitivity to ChatGPT "
    "and Gemini. However, this improvement was accompanied by a severe increase in false-positive "
    "rates on human text, particularly technical writing. Therefore, the retrained five-feature "
    "Logistic Regression model was not promoted to production."
)

CHATGPT_GENERIC_PARAGRAPH = (
    "Artificial intelligence has rapidly transformed industries across the globe. "
    "From healthcare to finance, AI-powered systems are revolutionizing how organizations "
    "operate and make decisions. Machine learning algorithms now process vast amounts of data "
    "with unprecedented accuracy, enabling businesses to gain deeper insights and optimize "
    "their operations. The integration of neural networks and deep learning has further "
    "accelerated this transformation, creating new possibilities for automation and intelligent "
    "decision-making. As AI continues to evolve, its applications become increasingly "
    "sophisticated, driving innovation and shaping the future of work across all sectors."
)


def _get_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


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


def _extract_single_sample(text: str) -> list:
    """Worker function for single text feature extraction."""
    sys.path.insert(0, str(_PROJ_ROOT))
    from text_forensics.signals.curvature import get_curvature
    from text_forensics.signals.burstiness import get_burstiness
    from text_forensics.signals.lexical_entropy import get_lexical_stats
    from text_forensics.signals.structural_regularity import get_structural_regularity
    from text_forensics.signals.cliche_scanner import get_cliche_density

    curv = get_curvature(text, model_name="HuggingFaceTB/SmolLM2-135M")
    burst = get_burstiness(text)
    ent = get_lexical_stats(text)["entropy"]
    struct = get_structural_regularity(text)["composite"]
    cliche = get_cliche_density(text)

    return [curv, burst, ent, struct, cliche]


def _worker_wrapper(args):
    idx, text = args
    try:
        row = _extract_single_sample(text)
        return idx, text, row, None
    except Exception as e:
        return idx, text, None, str(e)


def extract_all_features_parallel(samples: list[tuple[str, int]], num_workers: int = 4) -> tuple[np.ndarray, np.ndarray]:
    cache = {}
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info("Loaded %d existing cached feature records.", len(cache))
        except Exception as e:
            logger.warning("Cache load failed: %s", e)

    # Find tasks to compute
    pending_tasks = []
    for idx, (text, label) in enumerate(samples):
        key = _get_key(text)
        if key not in cache or len(cache[key]) != 5:
            pending_tasks.append((idx, text))

    logger.info("Total samples: %d | Cached: %d | To Extract: %d",
                len(samples), len(samples) - len(pending_tasks), len(pending_tasks))

    if pending_tasks:
        logger.info("Launching %d worker processes for parallel extraction...", num_workers)
        t0 = time.time()
        completed = 0
        total_pending = len(pending_tasks)

        with mp.Pool(processes=num_workers) as pool:
            for idx, text, row, err in pool.imap_unordered(_worker_wrapper, pending_tasks, chunksize=2):
                completed += 1
                key = _get_key(text)
                if row is not None:
                    cache[key] = row
                else:
                    logger.warning("Error extracting sample %d: %s", idx, err)

                if completed % 25 == 0 or completed == total_pending:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_pending - completed) / rate if rate > 0 else 0
                    logger.info("Progress: [%d/%d] (%.1f%%) | %.2f texts/s | ETA: %.1fs",
                                completed, total_pending, completed / total_pending * 100, rate, eta)
                    # Persist cache incrementally
                    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
                        json.dump(cache, f, indent=2)

    # Save final cache
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    # Build matrix in original order
    rows = []
    labels = []
    for text, label in samples:
        key = _get_key(text)
        row = cache.get(key, [None] * 5)
        rows.append(row)
        labels.append(label)

    return np.array(rows, dtype=object), np.array(labels, dtype=int)


def main():
    logger.info("=================================================================")
    logger.info("Starting Modern 5-Feature Model Training (SmolLM2-135M Curvature)")
    logger.info("=================================================================")

    # 1. Load data
    human_samples = _load_corpus(_CALIB_DIR / "genre_corpus_additions_human.json", label=0)
    human_samples += _load_corpus(_CALIB_DIR / "hc3_test_pool_human.json", label=0)

    ai_samples = _load_corpus(_CALIB_DIR / "genre_corpus_additions_ai.json", label=1)
    ai_samples += _load_corpus(_CALIB_DIR / "hc3_test_pool_ai.json", label=1)

    logger.info("Loaded %d Human samples and %d AI samples (Total: %d)",
                len(human_samples), len(ai_samples), len(human_samples) + len(ai_samples))

    # Deterministic split
    random.seed(RANDOM_SEED)
    random.shuffle(human_samples)
    random.shuffle(ai_samples)

    def _split(lst):
        n = len(lst)
        n_tr = int(n * TRAIN_RATIO)
        n_v = int(n * VAL_RATIO)
        return lst[:n_tr], lst[n_tr:n_tr + n_v], lst[n_tr + n_v:]

    h_tr, h_val, h_te = _split(human_samples)
    a_tr, a_val, a_te = _split(ai_samples)

    train_samples = h_tr + a_tr
    val_samples = h_val + a_val
    test_samples = h_te + a_te

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    all_samples = train_samples + val_samples + test_samples
    logger.info("Splits: Train=%d, Val=%d, Test=%d", len(train_samples), len(val_samples), len(test_samples))

    # 2. Extract features in parallel
    num_workers = min(6, os.cpu_count() or 4)
    X_raw, y_all = extract_all_features_parallel(all_samples, num_workers=num_workers)

    n_train = len(train_samples)
    n_val = len(val_samples)
    n_test = len(test_samples)

    X_train_raw = X_raw[:n_train]
    y_train = y_all[:n_train]

    X_val_raw = X_raw[n_train:n_train + n_val]
    y_val = y_all[n_train:n_train + n_val]

    X_test_raw = X_raw[n_train + n_val:]
    y_test = y_all[n_train + n_val:]

    # 3. Calculate feature column means on train set (for imputation)
    col_means = {}
    for col_idx, col_name in enumerate(FEATURE_ORDER):
        valid_vals = [r[col_idx] for r in X_train_raw if r[col_idx] is not None]
        mean_val = float(np.mean(valid_vals)) if valid_vals else 0.0
        col_means[col_name] = mean_val
        logger.info("Train Mean for '%s': %.4f", col_name, mean_val)

    def _impute(matrix):
        out = np.zeros(matrix.shape, dtype=float)
        for r_idx in range(matrix.shape[0]):
            for c_idx in range(matrix.shape[1]):
                val = matrix[r_idx, c_idx]
                if val is None or np.isnan(float(val)):
                    out[r_idx, c_idx] = col_means[FEATURE_ORDER[c_idx]]
                else:
                    out[r_idx, c_idx] = float(val)
        return out

    X_train = _impute(X_train_raw)
    X_val = _impute(X_val_raw)
    X_test = _impute(X_test_raw)

    # 4. Standard Scaler & Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(X_train_scaled, y_train)

    # 5. Evaluate on Test Set
    test_probs = clf.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_probs >= 0.50).astype(int)

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    auc = roc_auc_score(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    logger.info("================== TEST METRICS ==================")
    logger.info("Accuracy:  %.4f", acc)
    logger.info("Precision: %.4f", prec)
    logger.info("Recall:    %.4f", rec)
    logger.info("F1 Score:  %.4f", f1)
    logger.info("ROC-AUC:   %.4f", auc)
    logger.info("FPR:       %.4f (FP=%d, TN=%d)", fpr, fp, tn)
    logger.info("FNR:       %.4f (FN=%d, TP=%d)", fnr, fn, tp)
    logger.info("==================================================")

    # 6. Coefficients
    coef_dict = {f: float(c) for f, c in zip(FEATURE_ORDER, clf.coef_[0])}
    intercept = float(clf.intercept_[0])
    logger.info("Learned Coefficients: %s", coef_dict)
    logger.info("Intercept: %.4f", intercept)

    # 7. Compute updated baseline_stats (for CDF mappings) on human training distribution
    human_train_rows = [X_train_raw[i] for i in range(len(y_train)) if y_train[i] == 0]
    baseline_stats = {}
    for col_idx, col_name in enumerate(FEATURE_ORDER):
        vals = [float(r[col_idx]) for r in human_train_rows if r[col_idx] is not None]
        if vals:
            baseline_stats[col_name] = {
                "mu0": float(np.mean(vals)),
                "sigma0": float(np.std(vals)) or 1.0,
                "n_samples": len(vals),
            }

    # 8. Benchmark Evaluation
    def _evaluate_text(name: str, text: str):
        from text_forensics.feature_extractor import extract_five_features
        feat_dict = extract_five_features(text)
        vec = [feat_dict[k] for k in FEATURE_ORDER]
        vec_imputed = [col_means[k] if v is None else v for k, v in zip(FEATURE_ORDER, vec)]
        vec_scaled = scaler.transform(np.array(vec_imputed).reshape(1, -1))
        prob = float(clf.predict_proba(vec_scaled)[0, 1])
        if prob >= 0.70:
            verdict = "AI"
        elif prob >= 0.45:
            verdict = "Likely AI"
        elif prob <= 0.20:
            verdict = "Human"
        else:
            verdict = "Likely Human"
        logger.info("Benchmark [%s] -> P(AI)=%.4f (%s) | Features: %s", name, prob, verdict, feat_dict)
        return {"name": name, "prob": prob, "verdict": verdict, "features": {k: feat_dict[k] for k in FEATURE_ORDER}}

    bench_results = [
        _evaluate_text("CHATGPT_TECHNICAL", CHATGPT_TECHNICAL_PARAGRAPH),
        _evaluate_text("CHATGPT_GENERIC", CHATGPT_GENERIC_PARAGRAPH),
        _evaluate_text("HUMAN_PAXOS", HUMAN_PAXOS_PARAGRAPH),
        _evaluate_text("HUMAN_HISTORY", HUMAN_HISTORY_PARAGRAPH),
    ]

    # 9. Save Artifacts
    with open(_OUTPUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(_OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(_OUTPUT_DIR / "baseline_stats.json", "w", encoding="utf-8") as f:
        json.dump(baseline_stats, f, indent=2)

    metadata = {
        "model_name": "Modern 5-Feature Logistic Regression (SmolLM2-135M)",
        "surrogate_causal_model": "HuggingFaceTB/SmolLM2-135M",
        "feature_order": FEATURE_ORDER,
        "feature_means": col_means,
        "coefficients": coef_dict,
        "intercept": intercept,
        "thresholds_4way": {
            "human_max": 0.20,
            "likely_human_max": 0.45,
            "likely_ai_min": 0.45,
            "ai_min": 0.70,
        },
        "train_n": n_train,
        "val_n": n_val,
        "test_n": n_test,
        "random_seed": RANDOM_SEED,
        "test_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "benchmarks": bench_results,
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(_OUTPUT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved all model artifacts to %s", _OUTPUT_DIR)
    logger.info("Training and evaluation completed successfully.")


if __name__ == "__main__":
    main()
