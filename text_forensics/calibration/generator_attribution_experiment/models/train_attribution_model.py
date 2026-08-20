"""
text_forensics/calibration/generator_attribution_experiment/models/train_attribution_model.py

Trains, compares, and calibrates candidate classifiers for Stage-2 AI-Generator Attribution:
  Classes: ["ChatGPT", "Gemini", "Claude", "Other_AI"]

Strictly leakage-free:
  - Fits scaler and models ONLY on the Train split.
  - Compares candidates on the Validation split (unseen prompts).
  - Keeps the Test split completely untouched until final evaluation.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _HERE.parent
_ROOT = _EXPERIMENT_DIR.parent.parent.parent
sys.path.insert(0, str(_ROOT))

from text_forensics.calibration.generator_attribution_experiment.features.attribution_features import (
    ATTRIBUTION_FEATURE_ORDER,
    extract_attribution_features,
    extract_feature_vector,
)

_DATASET_PATH = _EXPERIMENT_DIR / "dataset" / "attribution_dataset.jsonl"
_MODELS_DIR = _EXPERIMENT_DIR / "models"
_SCALERS_DIR = _EXPERIMENT_DIR / "scalers"
_CONFIGS_DIR = _EXPERIMENT_DIR / "configs"

CLASSES = ["ChatGPT", "Gemini", "Claude", "Other_AI"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}


def load_and_extract_features() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[Dict]]]:
    """Loads dataset and extracts feature vectors for train, val, and test splits."""
    with open(_DATASET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Filter only AI samples for generator attribution (Human is handled at Stage 1)
    ai_records = [r for r in records if r["is_ai"] == 1]
    logger.info("Loaded %d AI records for attribution training", len(ai_records))

    cache_path = _EXPERIMENT_DIR / "dataset" / "feature_cache.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    split_X: Dict[str, List[List[float]]] = {"train": [], "val": [], "test": []}
    split_y: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    split_meta: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

    for r in ai_records:
        text = r["text"]
        gen = r["generator"]
        split = r["split"]

        if gen not in CLASS_TO_IDX:
            continue

        label_idx = CLASS_TO_IDX[gen]
        t_hash = str(hash(text[:200]))

        if t_hash in cache:
            feat_dict = cache[t_hash]
        else:
            feat_dict = extract_attribution_features(text)
            cache[t_hash] = feat_dict

        vec = extract_feature_vector(feat_dict)
        split_X[split].append(vec)
        split_y[split].append(label_idx)
        split_meta[split].append(r)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    X_arrays = {k: np.array(v, dtype=float) for k, v in split_X.items()}
    y_arrays = {k: np.array(v, dtype=int) for k, v in split_y.items()}

    for k in ["train", "val", "test"]:
        logger.info("Split '%s': X shape=%s, y shape=%s", k, X_arrays[k].shape, y_arrays[k].shape)

    return X_arrays, y_arrays, split_meta


def train_and_compare_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Any, StandardScaler, Dict[str, Any]]:
    """
    Fits scaler on X_train only, trains candidate classifiers, and selects
    the candidate with the highest Macro-F1 on the validation set.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    candidates = {
        "Multinomial_Logistic_Regression": LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient_Boosting": GradientBoostingClassifier(
            n_estimators=60,
            learning_rate=0.08,
            max_depth=3,
            random_state=42,
        ),
        "Extra_Trees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
        ),
    }

    comparison_results = {}
    best_model_name = None
    best_macro_f1 = -1.0
    best_model = None

    for name, clf in candidates.items():
        clf.fit(X_train_scaled, y_train)
        y_val_pred = clf.predict(X_val_scaled)
        
        acc = accuracy_score(y_val, y_val_pred)
        macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        
        logger.info("Candidate [%s] -> Val Accuracy: %.4f | Val Macro-F1: %.4f", name, acc, macro_f1)
        comparison_results[name] = {
            "val_accuracy": float(round(acc, 4)),
            "val_macro_f1": float(round(macro_f1, 4)),
        }

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_name = name
            best_model = clf

    logger.info("Winning candidate: [%s] with Val Macro-F1: %.4f", best_model_name, best_macro_f1)
    
    metadata = {
        "winning_candidate": best_model_name,
        "classes": CLASSES,
        "feature_order": ATTRIBUTION_FEATURE_ORDER,
        "n_features": len(ATTRIBUTION_FEATURE_ORDER),
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "candidate_comparison": comparison_results,
        "feature_means": {f: float(round(m, 6)) for f, m in zip(ATTRIBUTION_FEATURE_ORDER, scaler.mean_)},
        "feature_scales": {f: float(round(s, 6)) for f, m, s in zip(ATTRIBUTION_FEATURE_ORDER, scaler.mean_, scaler.scale_)},
        "uncertainty_thresholds": {
            "min_confidence_for_attribution": 0.45,
            "max_entropy_for_attribution": 1.30,
            "fallback_class": "Unknown / Other AI"
        }
    }

    return best_model, scaler, metadata


def save_artifacts(model: Any, scaler: StandardScaler, metadata: Dict[str, Any]):
    """Serializes the winning model, scaler, and metadata."""
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = _MODELS_DIR / "attribution_model.pkl"
    scaler_path = _SCALERS_DIR / "attribution_scaler.pkl"
    meta_path = _CONFIGS_DIR / "attribution_metadata.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved model -> %s", model_path)
    logger.info("Saved scaler -> %s", scaler_path)
    logger.info("Saved metadata -> %s", meta_path)


def main():
    logger.info("=== Starting Generator Attribution Model Training ===")
    X_arrays, y_arrays, _ = load_and_extract_features()
    best_model, scaler, metadata = train_and_compare_candidates(
        X_arrays["train"], y_arrays["train"],
        X_arrays["val"], y_arrays["val"]
    )
    save_artifacts(best_model, scaler, metadata)
    print("\nTraining completed successfully:")
    print(json.dumps(metadata["candidate_comparison"], indent=2))


if __name__ == "__main__":
    main()
