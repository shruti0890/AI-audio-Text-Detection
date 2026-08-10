"""
tests/test_transformer_article.py

Test runner for the Transformer Architecture technical test case.
Covers Correction 6, 7, and 8.
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
from text_forensics.signals.sentence_scorer import score_sentences

TRANSFORMER_TEST_TEXT = (
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
)


def run_test():
    print("=== Running analyze_text() on Transformer technical article ===")
    res = analyze_text(TRANSFORMER_TEST_TEXT)
    print(json.dumps(res, indent=2))

    print("\n=== Running score_sentences() (Correction 8) ===")
    sent_scores = score_sentences(TRANSFORMER_TEST_TEXT)
    print(json.dumps(sent_scores, indent=2))


if __name__ == "__main__":
    run_test()
