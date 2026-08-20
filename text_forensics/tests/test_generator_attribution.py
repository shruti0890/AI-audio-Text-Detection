"""
text_forensics/tests/test_generator_attribution.py

Unit and regression test suite for Stage-2 AI-Generator Attribution Module:
  1. Data leakage audit (grouped prompt split verification)
  2. 22-feature extraction schema verification
  3. Attribution model inference and probability normalization
  4. Two-stage conditional pipeline (Human bypass vs. AI attribution)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from text_forensics.calibration.generator_attribution_experiment.features.attribution_features import (
    ATTRIBUTION_FEATURE_ORDER,
    extract_attribution_features,
    extract_feature_vector,
)
from text_forensics.calibration.generator_attribution_experiment.attribution_pipeline import (
    analyze_text_with_attribution,
    load_attribution_engine,
)

_EXP_DIR = Path(__file__).resolve().parent.parent / "calibration" / "generator_attribution_experiment"


class TestAttributionDataIntegrity:
    """Verifies strict absence of data leakage in dataset partitioning."""

    def test_zero_prompt_overlap_between_splits(self):
        meta_path = _EXP_DIR / "configs" / "dataset_metadata.json"
        assert meta_path.exists(), "dataset_metadata.json missing"

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        train_p = set(meta["train_prompts"])
        val_p = set(meta["val_prompts"])
        test_p = set(meta["test_prompts"])

        assert len(train_p.intersection(val_p)) == 0, "Leakage detected between Train and Val prompts!"
        assert len(train_p.intersection(test_p)) == 0, "Leakage detected between Train and Test prompts!"
        assert len(val_p.intersection(test_p)) == 0, "Leakage detected between Val and Test prompts!"


class TestAttributionFeatureExtractor:
    """Verifies the 22-feature attribution extraction engine."""

    def test_feature_order_and_count(self):
        assert len(ATTRIBUTION_FEATURE_ORDER) == 22

    def test_feature_extraction_values(self):
        text = (
            "The Raft consensus algorithm decomposes distributed consensus into leader election, "
            "log replication, and safety guarantees. In Raft, nodes exist in one of three states: "
            "Follower, Candidate, or Leader."
        )
        feats = extract_attribution_features(text)
        assert isinstance(feats, dict)
        for fname in ATTRIBUTION_FEATURE_ORDER:
            assert fname in feats
            assert isinstance(feats[fname], (int, float))

        vec = extract_feature_vector(feats)
        assert len(vec) == 22


class TestAttributionModelInference:
    """Verifies model loading, inference, and probability calibration."""

    def test_model_loading_and_prediction(self):
        model, scaler, metadata = load_attribution_engine()
        assert model is not None, "Failed to load attribution model"
        assert scaler is not None, "Failed to load attribution scaler"
        assert metadata is not None, "Failed to load attribution metadata"

        dummy_vec = np.zeros((1, 22))
        dummy_scaled = scaler.transform(dummy_vec)
        probs = model.predict_proba(dummy_scaled)[0]

        assert len(probs) == 4
        assert np.isclose(np.sum(probs), 1.0, atol=1e-3)


class TestTwoStagePipelineIntegration:
    """Verifies conditional execution between Stage 1 and Stage 2."""

    def test_human_text_bypasses_attribution(self):
        human_text = (
            "Seems like a lot of work for very little game. You can only do it once or twice "
            "before they figure out what you are doing, and you'd have to come up with a way slash "
            "situation in which it was stolen without it looking like it was your fault. "
            "Which probably means smashing a window to make it look like a break-in. "
            "And a new window would cost more than what you make in selling the gun."
        )
        res = analyze_text_with_attribution(human_text, run_robustness=False)
        assert res["verdict"] in ["Human", "Likely Human"]
        assert res["generator_attribution"] is None
        assert res["predicted_generator"] == "Not applicable"

    def test_ai_text_triggers_attribution(self):
        ai_text = (
            "In Agentic AI, an agent is an AI system that can independently understand a goal, "
            "make decisions, plan steps, use tools, and take actions to achieve that goal. "
            "Unlike a normal AI model that mainly responds to a single prompt, an agent can break a complex task."
        )
        res = analyze_text_with_attribution(ai_text, run_robustness=False)
        assert res["verdict"] in ["Likely AI", "AI"]
        assert res["generator_attribution"] is not None
        assert "chatgpt" in res["generator_attribution"]
        assert "gemini" in res["generator_attribution"]
        assert "claude" in res["generator_attribution"]
        assert "other_ai" in res["generator_attribution"]
        assert isinstance(res["generator_confidence"], float)
        assert len(res["attribution_disclaimer"]) > 0
