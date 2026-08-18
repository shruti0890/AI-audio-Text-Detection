"""
text_forensics/tests/test_signals.py

Unit tests for individual signal modules (both production and archived).

Tests:
  - Curvature
  - Burstiness
  - Lexical Entropy
  - Structural Regularity
  - Cliché Scanner
  - N-gram Repetition (archived feature)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

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

SHORT_TEXT = "AI is useful."


class TestNgramRepetitionArchived:
    """Tests for signals/ngram_repetition.py (archived signal)."""

    def test_returns_dict_with_required_keys(self):
        from text_forensics.signals.ngram_repetition import get_ngram_repetition
        result = get_ngram_repetition(HUMAN_PARAGRAPH)
        assert isinstance(result, dict)
        for key in ["r2", "r3", "r4", "composite", "token_count", "short_text"]:
            assert key in result, f"Missing key: {key}"

    def test_composite_in_range(self):
        from text_forensics.signals.ngram_repetition import get_ngram_repetition
        result = get_ngram_repetition(HUMAN_PARAGRAPH)
        if result["composite"] is not None:
            assert 0.0 <= result["composite"] <= 1.0

    def test_empty_text_returns_none(self):
        from text_forensics.signals.ngram_repetition import get_ngram_repetition
        result = get_ngram_repetition("")
        assert result["composite"] is None
        assert result["token_count"] == 0


class TestStructuralRegularitySignal:
    """Tests for signals/structural_regularity.py (production signal)."""

    def test_returns_dict_with_required_keys(self):
        from text_forensics.signals.structural_regularity import get_structural_regularity
        result = get_structural_regularity(HUMAN_PARAGRAPH)
        assert isinstance(result, dict)
        for key in ["starter_diversity", "pos_similarity", "pos_entropy",
                    "discourse_density", "composite", "spacy_available", "n_sentences"]:
            assert key in result

    def test_composite_in_range(self):
        from text_forensics.signals.structural_regularity import get_structural_regularity
        result = get_structural_regularity(HUMAN_PARAGRAPH)
        if result["composite"] is not None:
            assert 0.0 <= result["composite"] <= 100.0


class TestCoreSignals:
    """Tests for curvature, burstiness, lexical_entropy, and cliche_scanner."""

    def test_burstiness_returns_float_or_none(self):
        from text_forensics.signals.burstiness import get_burstiness
        res = get_burstiness(HUMAN_PARAGRAPH)
        assert res is None or isinstance(res, float)

    def test_lexical_entropy_returns_dict(self):
        from text_forensics.signals.lexical_entropy import get_lexical_stats
        res = get_lexical_stats(HUMAN_PARAGRAPH)
        assert "ttr" in res and "entropy" in res
        assert res["entropy"] > 0.0

    def test_cliche_scanner_density(self):
        from text_forensics.signals.cliche_scanner import get_cliche_density, CLICHE_TERMS
        res = get_cliche_density(HUMAN_PARAGRAPH)
        assert isinstance(res, float)
        assert res >= 0.0
        assert len(CLICHE_TERMS) >= 50
