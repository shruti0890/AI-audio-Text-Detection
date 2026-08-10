"""
signals/cliche_scanner.py

Signal 3: Cliché & Buzzword Density.

References:
  - Kobak et al. 2024, "Delving into ChatGPT usage in academic writing across disciplines",
    arXiv:2403.07185. (Source for empirically-overrepresented words in ChatGPT output.)
  - Liang et al. 2024, "Mapping the Increasing Use of LLMs in Scientific Papers",
    arXiv:2404.01268. (Secondary source confirming high-frequency AI writing markers.)

The word list below is derived from those papers' frequency analyses and supplemented
with widely-observed community lists of ChatGPT-characteristic vocabulary. Each term
was chosen because it appears significantly more often in LLM-generated text than in
matched human corpora. The list is exposed as CLICHE_TERMS so it can be extended
without touching the detection logic.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cliché / AI-buzzword term list
# Sourced from: Kobak et al. 2024 (arXiv:2403.07185), Liang et al. 2024
# (arXiv:2404.01268), and widely-observed community compilations.
# 50 terms total (single words and short phrases).
# ---------------------------------------------------------------------------
CLICHE_TERMS: list[str] = [
    # Single-word markers (Kobak et al. empirically overrepresented in ChatGPT output)
    "delve",
    "delves",
    "delving",
    "pivotal",
    "testament",
    "tapestry",
    "beacon",
    "boasts",
    "underscores",
    "underscoring",
    "commendable",
    "multifaceted",
    "vibrant",
    "groundbreaking",
    "meticulous",
    "meticulously",
    "intricate",
    "intricacies",
    "leveraging",
    "revolutionize",
    "revolutionizing",
    "nuanced",
    "noteworthy",
    "embark",
    "embarking",
    "paramount",
    "synergy",
    "synergies",
    "robust",
    "holistic",
    "seamless",
    "seamlessly",
    "comprehensive",
    "esteemed",
    "innovative",
    "scalable",
    "transformative",
    "cutting-edge",
    "state-of-the-art",
    "unprecedented",
    # Phrase markers
    "in the realm of",
    "it is important to note",
    "it is worth noting",
    "a testament to",
    "in today's world",
    "in today's rapidly evolving",
    "foster innovation",
    "harness the power",
    "unlock the potential",
    "navigate the complexities",
]

# Sanity-check the list has at least 50 entries
assert len(CLICHE_TERMS) >= 50, (
    f"CLICHE_TERMS has only {len(CLICHE_TERMS)} entries; spec requires 50."
)

# Pre-compile one regex per term for efficiency
# Each pattern uses \b word boundaries (works correctly for single words;
# for multi-word phrases \b attaches to first/last word)
_COMPILED_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    for term in CLICHE_TERMS
]


def get_cliche_density(text: str) -> float:
    """
    Compute the cliché/buzzword density of the text as a percentage.

    density = (total_matches / total_words) × 100

    Each cliché term is counted independently. A single phrase like
    "in the realm of" counts as 1 match (not 4). Words not in CLICHE_TERMS
    do not contribute to the match count.

    Args:
        text: The input text to analyze.

    Returns:
        float: Density as a percentage [0.0, 100.0].
               Returns 0.0 for empty text or text with no alphabetic words.
    """
    if not text or not text.strip():
        return 0.0

    # Total word count: split on whitespace, count non-empty tokens
    words = text.split()
    total_words = len(words)

    if total_words == 0:
        return 0.0

    # Count matches across all patterns
    total_matches = 0
    for pattern in _COMPILED_PATTERNS:
        matches = pattern.findall(text)
        total_matches += len(matches)

    density = (total_matches / total_words) * 100.0
    return density
