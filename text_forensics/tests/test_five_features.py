"""
text_forensics/tests/test_five_features.py

Comprehensive test suite for the production Five-Feature Text Forensics Pipeline.

Tests:
  1. Five-feature extraction (features, details, keys)
  2. Exact canonical feature ordering (5 features)
  3. StandardScaler loading & transformation (5D)
  4. LogisticRegression loading & dimension verification (5 inputs)
  5. Probability output range [0.0, 1.0]
  6. Calibrated 4-way threshold classification logic
  7. Model_used identifier correctness
  8. Production model loading from five_feature_model/
  9. Fallback behavior when model is absent
  10. Short-text behavior (no crash, no NaN, no misleading extreme AI score)
  11. Sentence-level evidence aggregation
  12. Backward-compatible API keys
  13. Problematic AI technical paragraph regression test (P(AI) >= 0.70, AI verdict)
  14. Human technical paragraph evaluation
  15. New unseen AI technical paragraph evaluation

Run:
    pytest text_forensics/tests/test_five_features.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

HUMAN_PARAGRAPH = (
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

HUMAN_TECHNICAL_PARAGRAPH = (
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

PROBLEMATIC_AI_TECHNICAL_PARAGRAPH = (
    "Artificial intelligence has rapidly transformed industries across the globe. "
    "From healthcare to finance, AI-powered systems are revolutionizing how organizations "
    "operate and make decisions. Machine learning algorithms now process vast amounts of data "
    "with unprecedented accuracy, enabling businesses to gain deeper insights and optimize "
    "their operations. The integration of neural networks and deep learning has further "
    "accelerated this transformation, creating new possibilities for automation and intelligent "
    "decision-making. As AI continues to evolve, its applications become increasingly "
    "sophisticated, driving innovation and shaping the future of work across all sectors."
)

NEW_UNSEEN_AI_TECHNICAL_PARAGRAPH = (
    "In today's rapidly evolving digital landscape, artificial intelligence plays a pivotal role "
    "in revolutionizing modern enterprise workflows. By seamlessly integrating advanced neural "
    "architectures, organizations can unlock unprecedented efficiency across complex data pipelines. "
    "It is important to note that machine learning algorithms continually optimize decision-making "
    "processes to drive sustainable business value. Furthermore, these intelligent systems delve into "
    "intricate operational patterns to proactively mitigate operational risks. Ultimately, embracing "
    "these transformative technologies fosters innovation and empowers teams to stay ahead in a dynamic "
    "competitive environment."
)

SHORT_TEXT = "AI models process data."


# ---------------------------------------------------------------------------
# 1. Five-Feature Extraction & Canonical Ordering
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    """Tests for text_forensics/feature_extractor.py."""

    def test_extract_five_features_keys(self):
        from text_forensics.feature_extractor import extract_five_features, FEATURE_ORDER
        res = extract_five_features(HUMAN_PARAGRAPH, debug=False)
        assert isinstance(res, dict)
        for fname in FEATURE_ORDER:
            assert fname in res, f"Missing feature: {fname}"

    def test_feature_order_exact_five(self):
        from text_forensics.feature_extractor import FEATURE_ORDER
        expected = [
            "curvature",
            "burstiness",
            "lexical_entropy",
            "structural_regularity",
            "cliche_density",
        ]
        assert FEATURE_ORDER == expected, f"Feature order must be exact 5 features: {FEATURE_ORDER}"
        assert "ngram_repetition" not in FEATURE_ORDER, "ngram_repetition must NOT be in production FEATURE_ORDER"

    def test_build_feature_vector_length(self):
        from text_forensics.feature_extractor import extract_five_features, build_feature_vector
        feats = extract_five_features(HUMAN_PARAGRAPH, debug=False)
        vec = build_feature_vector(feats)
        assert len(vec) == 5, f"Feature vector must have exactly 5 elements, got {len(vec)}"

    def test_details_present(self):
        from text_forensics.feature_extractor import extract_five_features
        feats = extract_five_features(HUMAN_PARAGRAPH, debug=False)
        assert "_details" in feats
        d = feats["_details"]
        for key in ["curvature", "burstiness", "lexical_entropy", "structural_regularity", "cliche"]:
            assert key in d


# ---------------------------------------------------------------------------
# 2. Production Model and Scaler Artifacts Verification
# ---------------------------------------------------------------------------

class TestProductionArtifacts:
    """Verify five_feature_model/ artifacts and dimensions."""

    def test_artifacts_exist(self):
        model_dir = _PROJ_ROOT / "text_forensics" / "calibration" / "five_feature_model"
        assert (model_dir / "model.pkl").exists(), "model.pkl must exist in five_feature_model/"
        assert (model_dir / "scaler.pkl").exists(), "scaler.pkl must exist in five_feature_model/"
        assert (model_dir / "model_metadata.json").exists(), "model_metadata.json must exist"
        assert (model_dir / "evaluation_report.md").exists(), "evaluation_report.md must exist"

    def test_load_production_model_success(self):
        from text_forensics.fusion import load_five_feature_model
        model, scaler, meta = load_five_feature_model()
        assert model is not None, "Model must load successfully"
        assert scaler is not None, "Scaler must load successfully"
        assert meta is not None, "Metadata must load successfully"

    def test_model_and_scaler_dimensions(self):
        from text_forensics.fusion import load_five_feature_model
        import numpy as np
        model, scaler, meta = load_five_feature_model()

        # Scaler must have 5 features
        assert scaler.n_features_in_ == 5, f"Scaler expects 5 features, got {scaler.n_features_in_}"
        # Model must have 5 coefficients
        assert model.coef_.shape == (1, 5), f"Model coef shape must be (1, 5), got {model.coef_.shape}"

        # Test transformation with 5D dummy input
        dummy = np.zeros((1, 5))
        scaled = scaler.transform(dummy)
        proba = model.predict_proba(scaled)
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0][1] <= 1.0

    def test_metadata_records_production_settings(self):
        from text_forensics.fusion import load_five_feature_model
        _, _, meta = load_five_feature_model()
        assert meta["feature_order"] == [
            "curvature", "burstiness", "lexical_entropy",
            "structural_regularity", "cliche_density"
        ]
        assert meta.get("train_n", meta.get("partitions", {}).get("train_samples")) in {496, 508}
        assert meta.get("val_n", meta.get("partitions", {}).get("val_samples")) in {106, 121}
        assert meta.get("test_n", meta.get("partitions", {}).get("test_samples")) in {109, 152}
        assert meta.get("random_seed", meta.get("partitions", {}).get("random_seed")) == 42


# ---------------------------------------------------------------------------
# 3. Fusion and 4-Way Threshold Classification
# ---------------------------------------------------------------------------

class TestFusionClassification:
    """Tests for compute_five_feature_score()."""

    def test_probability_in_range(self):
        from text_forensics.fusion import compute_five_feature_score
        res = compute_five_feature_score(
            curvature=-0.5,
            burstiness=0.3,
            lexical_entropy=5.5,
            structural_regularity=30.0,
            cliche_density=1.5,
            baseline_stats={},
        )
        assert 0.0 <= res["ai_probability"] <= 1.0
        assert res["ai_score"] == round(res["ai_probability"] * 100.0, 2)
        assert res["model_used"] == "five_feature_logistic_regression"

    def test_four_way_threshold_boundaries(self):
        from text_forensics.fusion import compute_five_feature_score
        # Test boundary behavior
        # High AI input
        res_ai = compute_five_feature_score(
            curvature=2.5, burstiness=0.1, lexical_entropy=3.0,
            structural_regularity=80.0, cliche_density=5.0, baseline_stats={}
        )
        assert res_ai["verdict"] == "AI"
        assert res_ai["ai_probability"] >= 0.70

        # High Human input
        res_h = compute_five_feature_score(
            curvature=-2.5, burstiness=1.2, lexical_entropy=8.0,
            structural_regularity=10.0, cliche_density=0.0, baseline_stats={}
        )
        assert res_h["verdict"] == "Human"
        assert res_h["ai_probability"] <= 0.20

    def test_imputation_on_none_features(self):
        from text_forensics.fusion import compute_five_feature_score
        # Features with None (e.g. short text where burstiness is None)
        res = compute_five_feature_score(
            curvature=-0.8,
            burstiness=None,
            lexical_entropy=6.5,
            structural_regularity=None,
            cliche_density=0.0,
            baseline_stats={},
        )
        assert 0.0 <= res["ai_probability"] <= 1.0
        assert res["verdict"] in {"Human", "Likely Human", "Likely AI", "AI"}


# ---------------------------------------------------------------------------
# 4. Short Text and Robustness Handling
# ---------------------------------------------------------------------------

class TestShortTextHandling:
    """Verify short text does not crash, return NaN, or produce misleading extremes."""

    def test_short_text_pipeline(self):
        from text_forensics.pipeline import analyze_text
        res = analyze_text(SHORT_TEXT, run_robustness=False)
        assert res is not None
        assert 0.0 <= res["ai_probability"] <= 1.0
        assert res["short_text_warning"] is True
        assert res["features"]["burstiness"] is None
        assert res["verdict"] in {"Human", "Likely Human", "Likely AI", "AI"}

    def test_empty_text_raises(self):
        from text_forensics.pipeline import analyze_text
        with pytest.raises(ValueError):
            analyze_text("")
        with pytest.raises(ValueError):
            analyze_text("   ")


# ---------------------------------------------------------------------------
# 5. Sentence-Level Evidence
# ---------------------------------------------------------------------------

class TestSentenceScoringEvidence:
    """Verify sentence-level scoring and evidence aggregation."""

    def test_sentence_evidence_fields(self):
        from text_forensics.pipeline import analyze_text
        res = analyze_text(HUMAN_PARAGRAPH, run_robustness=False)
        se = res["sentence_evidence"]
        assert "mean_ai_probability" in se
        assert "upper_quartile" in se
        assert "ai_sentence_ratio" in se
        assert "n_sentences_analyzed" in se
        assert se["n_sentences_analyzed"] > 0


# ---------------------------------------------------------------------------
# 6. Backward-Compatible Output Schema
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Ensure all expected keys exist in analyze_text output."""

    def test_schema_keys(self):
        from text_forensics.pipeline import analyze_text
        res = analyze_text(HUMAN_PARAGRAPH, run_robustness=False)

        # Legacy PATH A keys
        assert "text_score" in res
        assert "signal_agreement" in res
        assert "signals" in res
        assert "stability_flag" in res
        assert "paraphrase_delta" in res

        # Production PATH B keys
        assert "ai_probability" in res
        assert "ai_score" in res
        assert "verdict" in res
        assert "confidence" in res
        assert "model_used" in res
        assert res["model_used"] == "five_feature_logistic_regression"
        assert "features" in res
        assert len(res["features"]) == 5
        assert "ngram_repetition" not in res["features"]


# ---------------------------------------------------------------------------
# 7. Regression Tests
# ---------------------------------------------------------------------------

class TestRegressionBenchmarks:
    """Regression tests on problematic AI paragraph, human text, and new AI text."""

    def test_problematic_ai_technical_paragraph(self):
        from text_forensics.pipeline import analyze_text
        res = analyze_text(PROBLEMATIC_AI_TECHNICAL_PARAGRAPH, run_robustness=False)
        # The 5-feature LR model must classify this polished technical text as AI
        assert res["ai_probability"] >= 0.70, (
            f"Problematic AI paragraph must be classified as AI (P >= 0.70), got P={res['ai_probability']}"
        )
        assert res["verdict"] == "AI", f"Verdict must be AI, got {res['verdict']}"

    def test_human_technical_paragraph(self):
        from text_forensics.pipeline import analyze_text
        res = analyze_text(HUMAN_TECHNICAL_PARAGRAPH, run_robustness=False)
        # Genuine human technical writing should not receive an extreme AI score
        assert res["ai_probability"] < 0.70, (
            f"Human technical text should not be classified as AI, got P={res['ai_probability']}"
        )
        assert res["verdict"] in {"Human", "Likely Human", "Likely AI"}

    def test_new_ai_technical_paragraph(self):
        from text_forensics.pipeline import analyze_text
        res = analyze_text(NEW_UNSEEN_AI_TECHNICAL_PARAGRAPH, run_robustness=False)
        assert res is not None
        assert res["ai_probability"] >= 0.45, (
            f"New AI technical text should be detected as AI/Likely AI, got P={res['ai_probability']}"
        )
        assert res["verdict"] in {"AI", "Likely AI"}

    def test_human_bicycle_trail_paragraph(self):
        from text_forensics.pipeline import analyze_text
        text = (
            "Trail is another factor bike designers take into account when talking about stability. "
            "Basically, Trail is the measurement between the point the fork of the bike is pointing at on the ground, "
            "and the point where the tire actually touches the ground. "
            "With the tire touching the ground behind where the fork is pointing, the front will act like a caster, "
            "like the wheels on the front of a shopping cart. It tends to straighten out when moving forward. "
            "However, again, this is in the full story. You can design a bike that has little to no trail, "
            "and that bike could still be perfectly stable."
        )
        res = analyze_text(text, run_robustness=False)
        assert res["ai_probability"] <= 0.55
        assert res["verdict"] in {"Human", "Likely Human"}
        assert res["ai_score"] == round(res["ai_probability"] * 100.0, 2)


# ---------------------------------------------------------------------------
# 8. UI Integration Consistency
# ---------------------------------------------------------------------------

class TestUiIntegrationConsistency:
    """Verify fusion_integration.run_full_pipeline provides consistent production probability and verdict."""

    def test_fusion_integration_text_consistency(self):
        from fusion_integration import run_full_pipeline
        text = (
            "Trail is another factor bike designers take into account when talking about stability. "
            "Basically, Trail is the measurement between the point the fork of the bike is pointing at on the ground, "
            "and the point where the tire actually touches the ground. "
            "With the tire touching the ground behind where the fork is pointing, the front will act like a caster, "
            "like the wheels on the front of a shopping cart. It tends to straighten out when moving forward. "
            "However, again, this is in the full story. You can design a bike that has little to no trail, "
            "and that bike could still be perfectly stable."
        )
        result = run_full_pipeline(text=text, run_robustness=False)
        assert result["text_score"] <= 55.0
        assert result["text_verdict"] in {"Human", "Likely Human"}
        assert result["text_ai_probability"] == round(result["text_score"] / 100.0, 4)
        assert "text_legacy_score" in result
        assert result["text_legacy_score"] != result["text_score"]
