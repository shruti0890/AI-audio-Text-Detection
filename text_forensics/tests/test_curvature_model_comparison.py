"""
tests/test_curvature_model_comparison.py

Correction 3: Benchmark distilgpt2 vs gpt2-medium on the same test inputs.

Tests:
  1. Phase 1.2 sanity test: 3 human + 3 AI paragraphs (same as original Phase 1.2).
  2. HC3 sample test: 3 human + 3 AI samples from the HC3 test pool (if available).

Reports:
  - Side-by-side raw curvature values for both models on each input.
  - CPU inference time per ~200-word input for each model.

Does NOT change the default model. Results are printed for user to decide.

Usage:
    python text_forensics/tests/test_curvature_model_comparison.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from text_forensics.signals.curvature import get_curvature

# ---------------------------------------------------------------------------
# Phase 1.2 sanity test texts (same as original Phase 1.2 report)
# ---------------------------------------------------------------------------
HUMAN_TEXTS = [
    # H1 — personal anecdote with irregular syntax
    (
        "My father was a man who rarely spoke about his feelings, which made the afternoon "
        "he sat me down in the kitchen to tell me he was proud of me one of the most startling "
        "experiences of my childhood. I was eleven. I had just finished reading a book he had "
        "given me — something about birds, I think — and he had watched me describe what I had "
        "learned with what I now recognize as his version of delight: very still, very quiet, "
        "a slight softening around his eyes."
    ),
    # H2 — descriptive prose with varied sentence lengths
    (
        "The ferry crossing from the mainland to the island takes forty minutes in good weather. "
        "Local passengers know to bring a jacket regardless of the season, because the temperature "
        "on the water bears little relation to the temperature on shore. Visitors rarely remember this. "
        "You can always spot them by the way they huddle against the railing, unprepared for the cold, "
        "staring at the receding harbor with an expression somewhere between surprise and betrayal."
    ),
    # H3 — reflective essay style
    (
        "Learning a second language in adulthood feels different from childhood acquisition. "
        "As an adult, you are always aware of the gap between what you are trying to say and what "
        "you are capable of saying. The gap closes over years, but it never quite disappears. "
        "Some speakers live with a permanent sense of operating just beneath their full expressive "
        "range — understanding more than they can produce, hearing the distance between what they "
        "intend and what comes out, adjusting for it constantly."
    ),
]

AI_TEXTS = [
    # A1 — corporate buzzword style
    (
        "In today's rapidly evolving landscape, organizations must leverage cutting-edge artificial "
        "intelligence solutions to unlock unprecedented value and foster innovation at scale. "
        "It is important to note that seamless integration of state-of-the-art machine learning "
        "algorithms is pivotal to driving transformative outcomes and navigating the complexities "
        "of a multifaceted global marketplace. By delving into the nuanced synergies between "
        "data-driven insights and holistic enterprise strategy, forward-thinking companies can "
        "harness the power of their digital ecosystems."
    ),
    # A2 — academic-adjacent AI style
    (
        "The realm of modern biotechnology boasts numerous commendable advancements that underscore "
        "the remarkable synergies between scientific innovation and medical progress. It is worth "
        "noting that these meticulous developments have seamlessly transformed the healthcare sector, "
        "enabling holistic solutions to some of the most nuanced challenges facing humanity today. "
        "The tapestry of genomic research, in particular, stands as a testament to the boundless "
        "potential of interdisciplinary collaboration across multiple domains."
    ),
    # A3 — structured explanatory AI style
    (
        "Artificial intelligence represents a beacon of transformative potential in the contemporary "
        "technological paradigm. It is paramount that organizations adopt a comprehensive and robust "
        "framework for integrating these state-of-the-art solutions into their existing workflows. "
        "The multifaceted nature of AI implementation requires meticulous attention to both the "
        "technical intricacies and the broader ethical considerations that underpin its responsible "
        "deployment at scale across modern enterprise ecosystems."
    ),
]


def benchmark_model(model_name: str, texts: list[tuple[str, str, str]]) -> list[dict]:
    """
    Run get_curvature() on all texts with the given model and time each call.

    Args:
        model_name: HuggingFace model identifier.
        texts: List of (label, text, kind) tuples.

    Returns:
        List of result dicts with keys: label, kind, score, elapsed_s.
    """
    results = []
    for label, text, kind in texts:
        t0 = time.time()
        try:
            score = get_curvature(text, model_name=model_name)
        except Exception as e:
            score = None
            print(f"    ERROR: {e}")
        elapsed = time.time() - t0
        results.append({
            "label": label,
            "kind": kind,
            "score": score,
            "elapsed_s": round(elapsed, 2),
        })
        score_str = f"{score:.4f}" if score is not None else "None"
        print(f"    {label} ({kind}): score={score_str}  [{elapsed:.1f}s]")
    return results


def run_comparison():
    """Run the full model comparison and print a report."""
    print("=" * 70)
    print("Correction 3: Curvature Model Comparison")
    print("  Models: distilgpt2  vs  gpt2-medium")
    print("=" * 70)

    # Build labeled text list
    texts = (
        [(f"H{i+1}", t, "human") for i, t in enumerate(HUMAN_TEXTS)] +
        [(f"A{i+1}", t, "AI") for i, t in enumerate(AI_TEXTS)]
    )

    # ---- Try to load HC3 test pool samples ----
    hc3_human_pool_path = Path(__file__).parent.parent / "calibration" / "hc3_test_pool_human.json"
    hc3_ai_pool_path    = Path(__file__).parent.parent / "calibration" / "hc3_test_pool_ai.json"
    hc3_texts: list[tuple[str, str, str]] = []

    if hc3_human_pool_path.exists() and hc3_ai_pool_path.exists():
        with open(hc3_human_pool_path, encoding="utf-8") as f:
            h_pool = json.load(f)
        with open(hc3_ai_pool_path, encoding="utf-8") as f:
            a_pool = json.load(f)
        # Take 3 from each (≈200 words each)
        for i, t in enumerate(h_pool[:3]):
            hc3_texts.append((f"HC3-H{i+1}", t, "human"))
        for i, t in enumerate(a_pool[:3]):
            hc3_texts.append((f"HC3-A{i+1}", t, "AI"))
        print(f"\nAdditional HC3 test pool: {len(hc3_texts)} samples loaded.")
    else:
        print("\nNote: HC3 test pool not yet available (run calibration first). Skipping HC3 samples.")

    all_texts = texts + hc3_texts

    results_by_model: dict[str, list[dict]] = {}
    for model_name in ["distilgpt2", "gpt2-medium"]:
        print(f"\n{'-' * 60}")
        print(f"Model: {model_name}")
        print(f"{'-' * 60}")
        results_by_model[model_name] = benchmark_model(model_name, all_texts)

    # ---- Side-by-side comparison table ----
    print("\n\n" + "=" * 70)
    print("Side-by-Side Comparison")
    print("=" * 70)
    print(f"{'Label':<12} {'Kind':<8} {'distilgpt2 score':>18} {'distilgpt2 t':>14} {'gpt2-medium score':>18} {'gpt2-medium t':>14}")
    print("-" * 90)

    distil = {r["label"]: r for r in results_by_model.get("distilgpt2", [])}
    medium = {r["label"]: r for r in results_by_model.get("gpt2-medium", [])}

    for label, _, kind in all_texts:
        d = distil.get(label, {})
        m = medium.get(label, {})
        d_score = f"{d['score']:.4f}" if d.get("score") is not None else "None"
        m_score = f"{m['score']:.4f}" if m.get("score") is not None else "None"
        d_t = f"{d.get('elapsed_s', '?'):.1f}s"
        m_t = f"{m.get('elapsed_s', '?'):.1f}s"
        print(f"  {label:<10} {kind:<8} {d_score:>18} {d_t:>14} {m_score:>18} {m_t:>14}")

    print("\n" + "-" * 70)
    print("Summary (averages):")
    for model_name in ["distilgpt2", "gpt2-medium"]:
        rs = results_by_model.get(model_name, [])
        human_scores = [r["score"] for r in rs if r["kind"] == "human" and r["score"] is not None]
        ai_scores    = [r["score"] for r in rs if r["kind"] == "AI"    and r["score"] is not None]
        times        = [r["elapsed_s"] for r in rs]
        h_mean = sum(human_scores) / len(human_scores) if human_scores else float("nan")
        a_mean = sum(ai_scores)    / len(ai_scores)    if ai_scores    else float("nan")
        t_mean = sum(times)        / len(times)        if times        else float("nan")
        print(f"  {model_name:<20}  human_mean={h_mean:.4f}  ai_mean={a_mean:.4f}  "
              f"separation={a_mean - h_mean:.4f}  avg_time={t_mean:.1f}s")
    print("=" * 70)
    print("\nNOTE: Higher separation (AI mean - human mean) = better discrimination.")
    print("Default model remains distilgpt2. Change via get_curvature(text, model_name='gpt2-medium').")


if __name__ == "__main__":
    run_comparison()
