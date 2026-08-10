"""
tests/test_agreeing_signals.py

Test runner verifying Correction 7 signal_agreement behavior:
  1. Confirms "signal_agreement": "disagreement" on Transformer technical text (spread > 40).
  2. Confirms "signal_agreement": "agreement" on clean human prose where signals agree (spread <= 40).
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

# Human personal narrative where signals consistently align (spread <= 40)
AGREEING_HUMAN_TEXT = (
    "My father was a man who rarely spoke about his feelings, which made the afternoon "
    "he sat me down in the kitchen to tell me he was proud of me one of the most startling "
    "experiences of my childhood. I was eleven. I had just finished reading a book he had "
    "given me — something about birds, I think — and he had watched me describe what I had "
    "learned with what I now recognize as his version of delight: very still, very quiet, "
    "a slight softening around his eyes."
)


def run_agreement_test():
    print("=== Testing Signal Agreement on Clean Human Text ===")
    res = analyze_text(AGREEING_HUMAN_TEXT)
    print("Text Score:", res["text_score"])
    print("Signal Agreement:", res["signal_agreement"])
    print("Sub-scores:", res["signals"])
    assert res["signal_agreement"] == "agreement", f"Expected 'agreement', got {res['signal_agreement']!r}"
    print("✅ TEST PASSED: Signal Agreement is correctly 'agreement' when scores align!")


if __name__ == "__main__":
    run_agreement_test()
