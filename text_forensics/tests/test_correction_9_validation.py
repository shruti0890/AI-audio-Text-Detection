"""
tests/test_correction_9_validation.py

Step 3 Validation Runner for Correction 9:
  Tests 4 real cases (Transformer AI Article, Human Narrative, Corporate AI, HC3 Human QA)
  under the rebalanced fusion config.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from text_forensics.pipeline import analyze_text

TEST_CASES = {
    "1_Transformer_AI_Article": (
        "The most important component of a transformer is attention, which can enable the model to "
        "determine the most important words in the text relative to other words in the same sentence. "
        "For example, in the sentence 'The animal didn't cross the street because it was too tired', "
        "the model can use attention to understand whether 'it' is referring to 'the animal' or 'the street'. "
        "This ability to capture relationships between words helps Transformers understand text much more effectively. "
        "A Transformer generally consists of two main components: an encoder and a decoder. The encoder "
        "processes the input sequence and creates meaningful representations of the information. "
        "The decoder uses these representations to generate the output sequence. Passing the Transformer "
        "architecture through multiple layers allows it to learn deeper features. However, "
        "unlike Recurrent Neural Networks (RNNs) or Long Short-Term Memory networks (LSTMs), which process "
        "sequences sequentially, Transformers process all tokens simultaneously. To preserve sequence order, "
        "positional encoding is added to the input embeddings. "
        "Another important concept of Transformer is self-attention. This mechanism allows a sequence "
        "to look at its own details and compute representations of itself rather than strictly "
        "one token after another; they create a mechanism to understand the context of each word within a sentence. "
        "Multi-head attention provides information about the context of tokens in different positions "
        "so that a single head does not get biased toward a specific feature. "
        "Transformers show several important advantages. First, their parallel processing capabilities "
        "speed up training significantly, making them more efficient than traditional recurrent architectures. "
        "Second, self-attention allows them to capture long-range dependencies between different parts "
        "of a sequence. Third, Transformers can be scaled to very large models and trained on massive "
        "datasets, which has contributed significantly to their state-of-the-art performance."
    ),
    "2_Human_Personal_Narrative": (
        "My father was a man who rarely spoke about his feelings, which made the afternoon "
        "he sat me down in the kitchen to tell me he was proud of me one of the most startling "
        "experiences of my childhood. I was eleven. I had just finished reading a book he had "
        "given me — something about birds, I think — and he had watched me describe what I had "
        "learned with what I now recognize as his version of delight: very still, very quiet, "
        "a slight softening around his eyes."
    ),
    "3_Corporate_AI_Text": (
        "In today's rapidly evolving digital landscape, organizations must leverage cutting-edge artificial "
        "intelligence solutions to unlock unprecedented value and foster innovation at scale. Seamless integration "
        "of state-of-the-art machine learning algorithms is pivotal to driving transformative outcomes and navigating "
        "the complexities of a multifaceted global marketplace. By delving into the nuanced synergies between data-driven "
        "insights and holistic enterprise strategy, forward-thinking companies can harness the power of their digital "
        "ecosystems to achieve sustainable and scalable growth in an ever-shifting paradigm."
    ),
    "4_HC3_Human_QA": (
        "The standard story of how the universe began is called the Big Bang theory. It states that all matter and space "
        "were compressed into an extremely dense and hot point that rapidly expanded about 13.8 billion years ago. "
        "Over billions of years, gravity pulled hydrogen and helium gas together to form the first stars and galaxies. "
        "Astronomers can still observe cosmic microwave background radiation today, which acts as an echo of this initial expansion."
    ),
}


def run_validation():
    print("=== Step 3 Validation Runner (Correction 9) ===")
    results = {}
    for name, text in TEST_CASES.items():
        res = analyze_text(text)
        score = res["text_score"]
        if score >= 70.0:
            verdict = "Likely AI-Generated"
        elif score >= 45.0:
            verdict = "Uncertain / Mixed Signals"
        else:
            verdict = "Likely Human-Written"
            
        results[name] = {
            "score": score,
            "verdict": verdict,
            "signal_agreement": res["signal_agreement"],
            "curvature_score": res["signals"]["curvature_score"],
            "entropy_score": res["signals"]["entropy_score"],
            "cliche_score": res["signals"]["cliche_score"],
        }
        print(f"\n{name}:")
        print(f"  Score: {score:.2f} / 100 -> Verdict: {verdict} (Agreement: {res['signal_agreement']})")
        print(f"  Signals -> Curvature: {res['signals']['curvature_score']}, Entropy: {res['signals']['entropy_score']}, Cliche: {res['signals']['cliche_score']}")

    with open(_SCRIPT_DIR / "correction_9_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_validation()
