"""
tests/test_abstract_breakdown.py

Analyzes the user's research paper abstract and prints the exact sub-score breakdown.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from text_forensics.pipeline import analyze_text
from text_forensics.signals.cliche_scanner import CLICHE_TERMS

ABSTRACT_TEXT = (
    "We consider the optimization of the Optimized Certainty Equivalent (OCE) risk, "
    "with applications including portfolio optimization in finance, and uncertainty quantification, "
    "classification, and regression in machine learning. Our contributions cover popular special cases of OCE, "
    "such as entropic risk, mean-variance risk, and smooth variants of Conditional Value-at-Risk. "
    "Our treatment sets out the conditions that facilitate the extension of OCE to unbounded r.v.s.. "
    "We provide a useful characterization of OCE that links OCE to utility-based shortfall risk (UBSR). "
    "Our characterization enables us to form an OCE estimator from the classic sample-average approximation (SAA) of UBSR. "
    "We derive mean-squared error (MSE) bounds for our proposed OCE estimator. For OCE optimization, "
    "we first derive an expression for the OCE gradient using the characterization linking OCE to UBSR. "
    "This expression serves as the basis for a gradient estimator for the OCE. We derive non-asymptotic bounds on the MSE "
    "for the proposed OCE gradient estimator. We incorporate the aforementioned gradient estimator into a stochastic gradient (SG) "
    "algorithm to optimize OCE and quantify its convergence rate using non-asymptotic bounds that we derive. "
    "Finally, we present three experiments that use our OCE optimization algorithm to solve portfolio optimization and uncertainty quantification problems."
)


def run():
    print("=== Analyzing Research Paper Abstract ===")
    res = analyze_text(ABSTRACT_TEXT)
    print(json.dumps(res, indent=2))

    matches = []
    for term in CLICHE_TERMS:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        found = pattern.findall(ABSTRACT_TEXT)
        if found:
            matches.extend(found)

    print("\nMatched Cliché/Buzzword Terms:", set(matches))


if __name__ == "__main__":
    run()
