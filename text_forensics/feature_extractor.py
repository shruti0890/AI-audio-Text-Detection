"""
text_forensics/feature_extractor.py

Five-Feature Extractor with Debug Mode (Production).

Extracts the 5 core production forensic features:
    1. curvature (Fast-DetectGPT via HuggingFaceTB/SmolLM2-135M)
    2. burstiness (σ/μ sentence length variation)
    3. lexical_entropy (Shannon entropy in bits)
    4. structural_regularity (starter diversity + POS overlap composite)
    5. cliche_density (50+ overused AI idioms %)

The canonical production feature ordering (used by Logistic Regression) is:
    ["curvature", "burstiness", "lexical_entropy", "structural_regularity", "cliche_density"]

This ordering is enforced in FEATURE_ORDER and must NEVER be changed.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical production feature ordering — 5 features
# ---------------------------------------------------------------------------
FEATURE_ORDER: List[str] = [
    "curvature",
    "burstiness",
    "lexical_entropy",
    "structural_regularity",
    "cliche_density",
]

# Project root for path resolution
_PROJ_ROOT = Path(__file__).resolve().parent.parent


def extract_five_features(text: str, debug: bool = False) -> Dict:
    """
    Extract the five core production forensic features from the given text.

    Args:
        text: Input text string.
        debug: If True, prints the full debug report to stdout.

    Returns:
        dict with keys:
            "curvature"            (float | None): raw Fast-DetectGPT discrepancy d(x)
            "burstiness"           (float | None): σ/μ sentence-length ratio (None if < 5 sents)
            "lexical_entropy"      (float | None): Shannon entropy in bits
            "structural_regularity"(float | None): structural composite [0, 100] (None if < 3 sents)
            "cliche_density"       (float):        cliché density %
            "_details"             (dict):         full subcomponent details for explainability
    """
    sys.path.insert(0, str(_PROJ_ROOT))

    from text_forensics.signals.curvature import get_curvature
    from text_forensics.signals.burstiness import get_burstiness
    from text_forensics.signals.lexical_entropy import get_lexical_stats
    from text_forensics.signals.structural_regularity import get_structural_regularity
    from text_forensics.signals.cliche_scanner import get_cliche_density, CLICHE_TERMS
    import re

    # ---- Feature 1: Curvature ----
    curvature_raw = get_curvature(text)

    # ---- Feature 2: Burstiness ----
    burstiness_raw = get_burstiness(text)

    # ---- Feature 3: Lexical Entropy ----
    lex_stats = get_lexical_stats(text)
    entropy_raw = lex_stats["entropy"]
    ttr_raw = lex_stats["ttr"]

    # ---- Feature 4: Structural Regularity ----
    struct_result = get_structural_regularity(text)
    struct_composite = struct_result["composite"]

    # ---- Feature 5: Cliché Density ----
    cliche_raw = get_cliche_density(text)
    matched_cliches = []
    for term in CLICHE_TERMS:
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        found = pattern.findall(text)
        matched_cliches.extend(found)

    features = {
        "curvature": curvature_raw,
        "burstiness": burstiness_raw,
        "lexical_entropy": entropy_raw,
        "structural_regularity": struct_composite,
        "cliche_density": cliche_raw,
    }

    details = {
        "curvature": {
            "raw": curvature_raw,
            "model": "HuggingFaceTB/SmolLM2-135M",
            "method": "Fast-DetectGPT analytical CDF",
        },
        "burstiness": {
            "raw": burstiness_raw,
            "formula": "σ/μ sentence-length ratio (requires >= 5 sentences)",
        },
        "lexical_entropy": {
            "raw": entropy_raw,
            "ttr": ttr_raw,
            "tokenization": "spaCy alphabetic tokens, lowercase",
        },
        "structural_regularity": {
            "starter_diversity": struct_result["starter_diversity"],
            "pos_similarity": struct_result["pos_similarity"],
            "pos_entropy": struct_result["pos_entropy"],
            "discourse_density": struct_result["discourse_density"],
            "composite": struct_composite,
            "spacy_available": struct_result["spacy_available"],
            "n_sentences": struct_result["n_sentences"],
        },
        "cliche": {
            "matched_phrases": matched_cliches,
            "density_pct": cliche_raw,
        },
    }

    if debug:
        _print_debug_report(features, details)

    return {**features, "_details": details}


# Backward-compatible alias
extract_six_features = extract_five_features


def build_feature_vector(features: Dict) -> List[Optional[float]]:
    """
    Build the canonical 5-element feature vector in FEATURE_ORDER.

    Args:
        features: dict from extract_five_features() (without "_details").

    Returns:
        list of 5 values in FEATURE_ORDER order. None values remain None.
    """
    return [features.get(f) for f in FEATURE_ORDER]


def _print_debug_report(features: Dict, details: Dict) -> None:
    """Print the full debug report."""
    print("\n" + "=" * 56)
    print("TEXT FORENSICS DEBUG (5-FEATURE PRODUCTION)")
    print("=" * 56)
    print("\nFeature values:\n")

    d = details

    # Curvature
    print("Curvature (Fast-DetectGPT):")
    print(f"    raw  = {d['curvature']['raw']}")
    print(f"    model= {d['curvature']['model']}")
    print()

    # Burstiness
    print("Burstiness:")
    print(f"    raw  = {d['burstiness']['raw']}")
    print(f"    note : {d['burstiness']['formula']}")
    print()

    # Lexical Entropy
    print("Lexical Entropy:")
    print(f"    raw (entropy) = {d['lexical_entropy']['raw']}")
    print(f"    raw (ttr)     = {d['lexical_entropy']['ttr']}")
    print(f"    tokenization  : {d['lexical_entropy']['tokenization']}")
    print()

    # Structural Regularity
    sr = d["structural_regularity"]
    print("Structural Regularity:")
    print(f"    starter diversity  = {sr['starter_diversity']}")
    print(f"    POS similarity     = {sr['pos_similarity']}")
    print(f"    POS entropy (bits) = {sr['pos_entropy']}")
    print(f"    discourse density  = {sr['discourse_density']}")
    print(f"    composite (0-100)  = {sr['composite']}")
    print(f"    spaCy available    = {sr['spacy_available']}")
    print(f"    sentences          = {sr['n_sentences']}")
    print()

    # Cliché
    cl = d["cliche"]
    print("Cliché Density:")
    print(f"    matched phrases = {cl['matched_phrases']}")
    print(f"    density %       = {cl['density_pct']}")
    print()

    print("=" * 56)
    print("FEATURE VECTOR (canonical 5-feature order)")
    print("=" * 56)
    for fname in FEATURE_ORDER:
        val = features.get(fname)
        print(f"  {fname:<28} = {val}")
    print()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Extract five forensic features from text (debug mode)."
    )
    parser.add_argument("text", nargs="?", help="Text to analyze (or use --file)")
    parser.add_argument("--file", type=str, help="Path to a .txt file to analyze")
    parser.add_argument("--debug", action="store_true", default=True, help="Print debug report")
    parser.add_argument("--json", action="store_true", help="Output features as JSON")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            input_text = fh.read()
    elif args.text:
        input_text = args.text
    else:
        print("Error: provide text as argument or use --file <path>", file=sys.stderr)
        sys.exit(1)

    result = extract_five_features(input_text, debug=args.debug)

    if args.json:
        out = {k: v for k, v in result.items() if k != "_details"}
        print(json.dumps(out, indent=2))
