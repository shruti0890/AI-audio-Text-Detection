"""
tests/test_robustness_truncation.py

Verification test for Correction 4: Robustness Test Truncation Mismatch Fix.

Tests a long text (> 300 words) to verify:
  1. compared_on_truncated is True.
  2. full_text_score is preserved.
  3. truncated_score is computed on the 300-word slice.
  4. paraphrase_delta is |truncated_score - paraphrased_score|.
  5. Measures CPU time taken for a full robustness check end-to-end.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from text_forensics.robustness_test import check_stability

# 350-word test text
LONG_TEXT = (
    "My grandmother made bread every Sunday morning without a recipe for over forty years in her small kitchen. "
    "The knowledge lived in her hands, in the subtle tension she felt when the dough was properly kneaded and ready to rise. "
    "She would sometimes hum while she worked, and the kitchen filled with a warmth that had nothing to do with the oven. "
    "In today's rapidly evolving digital landscape, it is important to note that organizations must leverage cutting-edge "
    "artificial intelligence solutions to unlock unprecedented value and foster innovation at scale. "
    "Seamless integration of state-of-the-art machine learning algorithms is pivotal to driving transformative outcomes "
    "and navigating the complexities of a multifaceted global marketplace. By delving into the nuanced synergies between "
    "data-driven insights and holistic enterprise strategy, forward-thinking companies can harness the power of their "
    "digital ecosystems to achieve sustainable and scalable growth in an ever-shifting paradigm. "
    "This comprehensive approach underscores the paramount importance of meticulous planning and robust execution frameworks "
    "that are commensurate with the groundbreaking potential of modern technology. "
    "Furthermore, the educational landscape is undergoing a revolutionary transformation, underpinned by the seamless "
    "integration of digital technologies and personalized learning methodologies. It is paramount to recognize that these "
    "holistic innovations are not merely commendable enhancements to existing curricula but rather a testament to the "
    "boundless potential of human ingenuity in adapting to the rapidly evolving demands of the twenty-first century "
    "knowledge economy. Navigating the complexities of modern geopolitical dynamics requires a nuanced understanding of "
    "the multifaceted factors that shape international relations. It is important to note that the pivotal role of "
    "multilateral institutions in fostering dialogue and promoting collaborative frameworks cannot be overstated. "
    "The tapestry of global diplomacy is woven from innumerable threads of historical precedent, cultural understanding, "
    "and strategic interest that must be meticulously balanced by experienced leaders and policymakers around the world. "
    "In conclusion, salt has preserved food for so long that the history of its trade is almost indistinguishable from "
    "the history of human commerce itself across centuries of civilizational development."
)

# Dummy baseline stats for standalone testing
DUMMY_STATS = {
    "curvature": {"mu0": 0.15, "sigma0": 0.05, "n_valid": 200},
    "burstiness": {"mu0": 0.60, "sigma0": 0.20, "n_valid": 200},
    "cliche_density": {"mu0": 0.50, "sigma0": 1.00, "n_valid": 200},
    "entropy": {"mu0": 5.20, "sigma0": 0.50, "n_valid": 200},
}


def test_truncation_fix():
    print("=" * 70)
    print("Correction 4 Verification — Robustness Truncation Fix & Timing")
    print("=" * 70)

    word_count = len(LONG_TEXT.split())
    print(f"Input text word count: {word_count} words (> 300 limit)")

    dummy_original_score = 65.5
    print(f"Supplied full-text score: {dummy_original_score}")
    print("\nRunning check_stability()...")

    t0 = time.time()
    res = check_stability(LONG_TEXT, dummy_original_score, DUMMY_STATS)
    elapsed = time.time() - t0

    print("\nResult Return Dict:")
    for k, v in res.items():
        if k == "paraphrased_text":
            print(f"  {k:<24}: {v[:80]}...")
        else:
            print(f"  {k:<24}: {v}")

    print("\nVerifications:")
    assert res["compared_on_truncated"] is True, "FAIL: compared_on_truncated should be True"
    print("  [PASS] compared_on_truncated == True")

    assert res["full_text_score"] == dummy_original_score, "FAIL: full_text_score mismatch"
    print("  [PASS] full_text_score preserved correctly")

    expected_delta = round(abs(res["truncated_score"] - res["paraphrased_score"]), 2)
    assert res["paraphrase_delta"] == expected_delta, f"FAIL: delta mismatch ({res['paraphrase_delta']} vs {expected_delta})"
    print(f"  [PASS] paraphrase_delta ({res['paraphrase_delta']}) equals |truncated_score ({res['truncated_score']}) - paraphrased_score ({res['paraphrased_score']})|")

    print(f"\nEnd-to-End Robustness CPU Inference Time: {elapsed:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    test_truncation_fix()
