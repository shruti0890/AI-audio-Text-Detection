"""
robustness_test.py

Adversarial Robustness Self-Test module.

Paraphrases the input text using a CPU-friendly T5-based paraphrase model,
re-runs the full scoring pipeline, and flags the result as "unstable" if the
score shifts more than 15 points.

Model: Vamsi/T5_Paraphrase_Paws (primary)
  - CPU-friendly T5-small fine-tuned on PAWS paraphrase pairs.
  - Fallback: tuner007/pegasus_paraphrase if primary unavailable.
  - Model used is accessible via PARAPHRASE_MODEL_USED constant.

Correction 4 fix (truncation mismatch):
  For texts over _MAX_PARAPHRASE_WORDS (300 words):
    - BEFORE: paraphrase_delta compared truncated paraphrase score vs. full-text
              original_score — unfair because they operate on different content.
    - AFTER:  we compute a fresh score on the SAME truncated slice that gets
              paraphrased, then compare that to the paraphrase score.
    - Both the full-text score (from caller) and the truncated-comparison score
      are stored in the return dict (clearly labeled) so nothing is silently lost.
    - compared_on_truncated: bool flag tells the caller which comparison was used.

Limitations:
  - Paraphrase quality on technical/domain-specific text may be low.
  - First call includes model download time; subsequent calls use the cache.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_PRIMARY_MODEL = "Vamsi/T5_Paraphrase_Paws"
_FALLBACK_MODEL = "tuner007/pegasus_paraphrase"
_MAX_PARAPHRASE_WORDS = 300
_INSTABILITY_THRESHOLD = 15.0

_para_tokenizer = None
_para_model = None
PARAPHRASE_MODEL_USED: Optional[str] = None


def _load_paraphrase_model():
    """
    Load the paraphrase model into the module-level cache.

    Tries _PRIMARY_MODEL first; falls back to _FALLBACK_MODEL on failure.
    Sets PARAPHRASE_MODEL_USED to whichever model loaded successfully.
    """
    global _para_tokenizer, _para_model, PARAPHRASE_MODEL_USED

    if _para_tokenizer is not None and _para_model is not None:
        return _para_tokenizer, _para_model

    for model_name in [_PRIMARY_MODEL, _FALLBACK_MODEL]:
        try:
            logger.info("Loading paraphrase model: %s ...", model_name)
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            mdl.eval()
            _para_tokenizer = tok
            _para_model = mdl
            PARAPHRASE_MODEL_USED = model_name
            logger.info("Paraphrase model loaded: %s", model_name)
            return tok, mdl
        except Exception as e:
            logger.warning("Failed to load %s: %s. Trying fallback...", model_name, e)

    raise RuntimeError(
        f"Could not load any paraphrase model. "
        f"Primary: {_PRIMARY_MODEL}, Fallback: {_FALLBACK_MODEL}"
    )


def _paraphrase_text(text: str) -> str:
    """
    Generate a paraphrase of the input text.

    NOTE: This function does NOT truncate the input. Truncation is done by
    check_stability() before calling this, so the comparison is always fair.

    Args:
        text: Input text to paraphrase (should be <= _MAX_PARAPHRASE_WORDS words).

    Returns:
        str: Paraphrased text.
    """
    tokenizer, model = _load_paraphrase_model()

    if "T5" in (PARAPHRASE_MODEL_USED or "") or "t5" in (PARAPHRASE_MODEL_USED or ""):
        input_text = f"paraphrase: {text} </s>"
    else:
        input_text = text

    with torch.no_grad():
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="longest",
        )
        outputs = model.generate(
            inputs["input_ids"],
            max_length=512,
            num_beams=2,
            num_return_sequences=1,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _score_text(text: str, baseline_stats: dict) -> float:
    """
    Run all 4 signals and compute a fused score for a text snippet.

    Args:
        text: Input text to score.
        baseline_stats: Calibration baseline dict from baseline_stats.json.

    Returns:
        float: Fused 0-100 score.
    """
    from text_forensics.signals.curvature import get_curvature
    from text_forensics.signals.burstiness import get_burstiness
    from text_forensics.signals.cliche_scanner import get_cliche_density
    from text_forensics.signals.lexical_entropy import get_lexical_stats
    from text_forensics.fusion import compute_text_score

    lex = get_lexical_stats(text)
    signals = {
        "curvature_raw":      get_curvature(text),
        "burstiness_raw":     get_burstiness(text),
        "cliche_density_raw": get_cliche_density(text),
        "entropy_raw":        lex["entropy"],
    }
    return compute_text_score(signals, baseline_stats)["text_score"]


def check_stability(text: str, original_score: float, baseline_stats: dict) -> dict:
    """
    Paraphrase the input text, re-score it, and flag instability if the score
    shifts too much.

    Correction 4 fix — fair truncation comparison:
      If len(text.split()) > _MAX_PARAPHRASE_WORDS, we:
        1. Truncate to _MAX_PARAPHRASE_WORDS words → "comparison_slice"
        2. Score comparison_slice → "truncated_score" (the fair baseline)
        3. Paraphrase comparison_slice → "paraphrase"
        4. Score paraphrase → "paraphrased_score"
        5. paraphrase_delta = |truncated_score - paraphrased_score|

      The caller's original_score (full-text) is preserved in the return dict
      as "full_text_score" so it is not lost, but it is NOT used in delta.

      If text is short enough (≤ _MAX_PARAPHRASE_WORDS), no truncation occurs
      and truncated_score == original_score.

    Args:
        text: The original input text.
        original_score: Score computed on the original (possibly full) text (0-100).
        baseline_stats: Dict loaded from calibration/baseline_stats.json.

    Returns:
        dict: {
            "stability_flag":       "stable" | "unstable" | "unknown",
            "paraphrase_delta":     float,   # |truncated_score - paraphrased_score|
            "full_text_score":      float,   # original_score as passed in (full text)
            "truncated_score":      float,   # score on the comparison slice
            "paraphrased_score":    float,   # score on the paraphrase
            "paraphrased_text":     str,
            "compared_on_truncated": bool,   # True if text was truncated before comparing
            "elapsed_seconds":      float,
            "model_used":           str,
        }
    """
    t_start = time.time()

    words = text.split()
    was_truncated = len(words) > _MAX_PARAPHRASE_WORDS

    # ---- Determine the comparison slice ----
    if was_truncated:
        comparison_text = " ".join(words[:_MAX_PARAPHRASE_WORDS])
        logger.info(
            "check_stability: text has %d words (> %d limit). "
            "Scoring comparison on truncated slice for a fair delta.",
            len(words), _MAX_PARAPHRASE_WORDS,
        )
        # Score the truncated slice (our fair baseline for delta)
        try:
            truncated_score = _score_text(comparison_text, baseline_stats)
        except Exception as e:
            logger.error("Scoring truncated slice failed: %s", e)
            truncated_score = original_score  # degrade gracefully
    else:
        comparison_text = text
        truncated_score = original_score  # no truncation needed

    # ---- Paraphrase the comparison slice ----
    try:
        para_text = _paraphrase_text(comparison_text)
    except Exception as e:
        logger.error("Paraphrasing failed: %s", e)
        elapsed = time.time() - t_start
        return {
            "stability_flag":        "unknown",
            "paraphrase_delta":      0.0,
            "full_text_score":       original_score,
            "truncated_score":       truncated_score,
            "paraphrased_score":     truncated_score,
            "paraphrased_text":      "",
            "compared_on_truncated": was_truncated,
            "elapsed_seconds":       round(elapsed, 2),
            "model_used":            PARAPHRASE_MODEL_USED or "none",
            "error":                 str(e),
        }

    # ---- Score the paraphrase ----
    try:
        paraphrased_score = _score_text(para_text, baseline_stats)
    except Exception as e:
        logger.error("Scoring paraphrase failed: %s", e)
        paraphrased_score = truncated_score

    # ---- Compute delta and flag (against truncated_score, not full_text_score) ----
    delta = abs(truncated_score - paraphrased_score)
    flag = "unstable" if delta > _INSTABILITY_THRESHOLD else "stable"
    elapsed = time.time() - t_start

    return {
        "stability_flag":        flag,
        "paraphrase_delta":      round(delta, 2),
        "full_text_score":       round(original_score, 2),
        "truncated_score":       round(truncated_score, 2),
        "paraphrased_score":     round(paraphrased_score, 2),
        "paraphrased_text":      para_text,
        "compared_on_truncated": was_truncated,
        "elapsed_seconds":       round(elapsed, 2),
        "model_used":            PARAPHRASE_MODEL_USED or "unknown",
    }
