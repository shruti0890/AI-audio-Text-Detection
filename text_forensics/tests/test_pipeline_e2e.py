"""
tests/test_pipeline_e2e.py

Phase 1.9 End-to-End Integration Test.

Calls analyze_text() on 5 varied inputs and prints the full output for each.
All five inputs cover the key test cases specified in Phase 1.9.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from text_forensics.pipeline import analyze_text

# ---------------------------------------------------------------------------
# Test inputs
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "label": "1. Human text (personal anecdote)",
        "text": (
            "My father was a man who rarely spoke about his feelings, which made the afternoon "
            "he sat me down in the kitchen to tell me he was proud of me one of the most startling "
            "experiences of my childhood. I was eleven. I had just finished reading a book he had "
            "given me — something about birds, I think — and he had watched me describe what I had "
            "learned with what I now recognize as his version of delight: very still, very quiet, "
            "a slight softening around his eyes. The pride came out at the end of the conversation "
            "almost as an afterthought. He said it and then stood up and went back to whatever he "
            "had been doing before. I carried those words around for years without telling anyone."
        ),
    },
    {
        "label": "2. AI text (corporate buzzword style)",
        "text": (
            "In today's rapidly evolving landscape, organizations must leverage cutting-edge "
            "artificial intelligence solutions to unlock unprecedented value and foster innovation "
            "at scale. It is important to note that seamless integration of state-of-the-art "
            "machine learning algorithms is pivotal to driving transformative outcomes and "
            "navigating the complexities of a multifaceted global marketplace. By delving into "
            "the nuanced synergies between data-driven insights and holistic enterprise strategy, "
            "forward-thinking companies can harness the power of their digital ecosystems to "
            "achieve sustainable, scalable growth and remain competitive in an ever-shifting "
            "paradigm. This comprehensive approach underscores the paramount importance of "
            "meticulous planning and robust execution frameworks that are commensurate with the "
            "groundbreaking potential of modern artificial intelligence."
        ),
    },
    {
        "label": "3. Very short text (< 50 words)",
        "text": "Hello. This is a short text for testing purposes.",
    },
    {
        "label": "4. Mixed text (human + AI spliced together)",
        "text": (
            "My grandmother made bread every Sunday morning without a recipe — the knowledge "
            "lived in her hands, in the tension she felt when the dough was properly kneaded. "
            "She would sometimes hum while she worked, and the kitchen filled with a warmth that "
            "had nothing to do with the oven. "
            "In today's rapidly evolving landscape, it is important to note that leveraging "
            "cutting-edge artificial intelligence is pivotal to driving transformative outcomes. "
            "Organizations must delve into the multifaceted synergies between data ecosystems "
            "and holistic enterprise strategies to harness the power of seamless integration."
        ),
    },
    {
        "label": "5. Non-English text (French)",
        "text": (
            "La Tour Eiffel a été construite entre 1887 et 1889 comme porte d'entrée pour "
            "l'Exposition universelle de Paris. Initialement critiquée par certains artistes "
            "et intellectuels parisiens, elle est devenue l'un des monuments les plus visités "
            "au monde et un symbole reconnu de la France. Sa structure en fer forgé a inspiré "
            "de nombreux ingénieurs et architectes dans les décennies qui ont suivi sa construction."
        ),
    },
]

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

def run_e2e_tests():
    """Run all 5 test cases through the full pipeline and print results."""
    print("=" * 80)
    print("Text Forensics Pipeline — End-to-End Integration Test (Phase 1.9)")
    print("=" * 80)

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n{'─' * 60}")
        print(f"Test {i}: {tc['label']}")
        print(f"Input ({len(tc['text'].split())} words): {tc['text'][:120]}{'...' if len(tc['text']) > 120 else ''}")
        print()

        try:
            result = analyze_text(tc["text"])
            print("Result:")
            print(json.dumps(result, indent=2, default=str))
        except (TypeError, ValueError) as e:
            print(f"[Expected exception]: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"[Unexpected error]: {type(e).__name__}: {e}")

    # ---- Error-case tests ----
    print(f"\n{'─' * 60}")
    print("Error case A: Empty string")
    try:
        analyze_text("")
    except ValueError as e:
        print(f"  ✅ ValueError raised: {e}")

    print()
    print("Error case B: Non-string input")
    try:
        analyze_text(42)  # type: ignore
    except TypeError as e:
        print(f"  ✅ TypeError raised: {e}")

    print()
    print("=" * 80)
    print("E2E test complete.")
    print("=" * 80)


if __name__ == "__main__":
    run_e2e_tests()
