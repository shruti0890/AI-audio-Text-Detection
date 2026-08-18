"""
signals/structural_regularity.py

Signal 5 (NEW): Sentence & Structural Regularity.

Combines four subcomponents into a single structural feature that captures
whether a document repeatedly uses highly similar sentence construction patterns.

This is a composite feature. Its subcomponents are:

    A. Sentence Starter Diversity
       Ratio of unique first-meaningful-words to number of sentences.
       Low diversity → AI-like (e.g. every sentence starts with "The model...").

    B. POS Pattern Similarity
       Extracts the POS tag sequence for each sentence (using spaCy).
       Computes what fraction of POS bigrams are shared across sentences —
       higher sharing → more structurally uniform → more AI-like.

    C. POS Distribution Entropy  [internal component, not a separate top-level feature]
       Shannon entropy of POS-tag distribution. Used internally to weight the composite.

    D. Discourse-Marker Density
       Density of discourse/transition markers per 100 tokens.

IMPORTANT scientific note:
    None of these subcomponents is a definitive AI indicator by itself.
    - Humans use transition words in academic writing.
    - Some human writing has low starter diversity by topic.
    - The direction of each signal is LEARNED by the Logistic Regression model.
    This feature is purely a forensic observation; it does not definitively
    prove AI authorship.

Research basis:
    - Stylometric features (POS distribution, sentence starters) are used
      throughout AI-generated text detection literature.
    - See: Guo et al. 2023 "How Close is ChatGPT to Human Experts?";
      Uchendu et al. 2021 "TURINGBENCH"; Zhu et al. 2023 "Beat LLMs at Their
      Own Game: Zero-Shot LLM-Generated Text Detection via Querying ChatGPT".
    - The specific composite formula is our project-specific engineering choice.

Implementation notes:
    - Uses spaCy en_core_web_sm (already a project dependency).
    - Falls back gracefully to non-spaCy computation if spaCy unavailable.
    - Returns a dict of subcomponent values for full auditability.
    - Final composite is in [0.0, 100.0] — higher → more AI-like (structurally
      uniform), SUBJECT TO CALIBRATION (the LR model may find the opposite).
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Discourse marker list (configurable)
# ---------------------------------------------------------------------------
# These are common discourse/transition markers. Inclusion here does NOT imply
# they are proof of AI authorship — they are common in academic human writing.
# The feature measures their DENSITY; the direction of that signal is learned.
DISCOURSE_MARKERS: list[str] = [
    "however",
    "therefore",
    "moreover",
    "furthermore",
    "additionally",
    "overall",
    "consequently",
    "in addition",
    "as a result",
    "for this reason",
    "thus",
    "although",
    "while",
    "in contrast",
    "on the other hand",
    "instead",
    "nevertheless",
    "nonetheless",
    "in summary",
    "in conclusion",
    "in particular",
    "specifically",
    "notably",
    "importantly",
    "indeed",
    "in fact",
    "for example",
    "for instance",
    "that is",
    "in other words",
    "as such",
    "given that",
    "it follows that",
    "this suggests",
    "this indicates",
    "this demonstrates",
]

# Pre-compile discourse marker patterns
_DISCOURSE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b" + re.escape(m) + r"\b", re.IGNORECASE)
    for m in DISCOURSE_MARKERS
]

# Minimum sentences for a reliable structural signal
_MIN_SENTENCES = 3

# Lazy-loaded spaCy model
_NLP = None


def _get_spacy() -> Optional[object]:
    """Load spaCy en_core_web_sm model (lazy, cached). Returns None on failure."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(
                "structural_regularity: Could not load spaCy en_core_web_sm: %s. "
                "POS-based subcomponents will be skipped.",
                e,
            )
            _NLP = False
    return _NLP if _NLP is not False else None


def _sentence_split(text: str) -> list[str]:
    """Split text into sentences using NLTK (falls back to period split)."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except Exception:
        return [s.strip() for s in text.split(".") if s.strip()]


def _first_meaningful_word(sentence: str) -> Optional[str]:
    """
    Extract the first alphabetic word from a sentence (lowercased).
    Returns None if no alphabetic word found.
    """
    for token in sentence.split():
        word = re.sub(r"[^a-zA-Z]", "", token)
        if word:
            return word.lower()
    return None


def _compute_starter_diversity(sentences: list[str]) -> float:
    """
    Compute sentence starter diversity.

        starter_diversity = unique_starters / total_sentences

    Returns float in [0.0, 1.0]. Higher → more diverse starters.

    Args:
        sentences: List of sentence strings.

    Returns:
        float: Starter diversity ratio.
    """
    starters = [_first_meaningful_word(s) for s in sentences]
    starters = [w for w in starters if w is not None]
    if not starters:
        return 1.0  # Fallback: neutral
    unique = len(set(starters))
    return unique / len(starters)


def _compute_pos_similarity(sentences: list[str], nlp) -> float:
    """
    Compute POS pattern similarity across sentences.

    For each sentence, extract its POS sequence as a list of tags (e.g. ["DET", "NOUN", "VERB"]).
    Then extract all POS bigrams. Compute the fraction of POS bigrams that are shared
    (appear in >1 sentence's POS sequence).

        POS_similarity = (repeated_unique_pos_bigrams) / (all_unique_pos_bigrams)

    Higher → sentences share more POS structure → more uniform (potentially AI-like).

    Args:
        sentences: List of sentence strings.
        nlp: spaCy Language model.

    Returns:
        float: POS pattern similarity in [0.0, 1.0].
    """
    all_bigrams_per_sentence: list[list[tuple]] = []

    for sent in sentences:
        doc = nlp(sent)
        pos_tags = [token.pos_ for token in doc if not token.is_space and not token.is_punct]
        if len(pos_tags) < 2:
            continue
        bigrams = [(pos_tags[i], pos_tags[i + 1]) for i in range(len(pos_tags) - 1)]
        all_bigrams_per_sentence.append(bigrams)

    if not all_bigrams_per_sentence:
        return 0.0

    # Global pool of POS bigrams and their sentence-occurrence counts
    bigram_sentence_counts: Counter = Counter()
    for i, sent_bigrams in enumerate(all_bigrams_per_sentence):
        for bg in set(sent_bigrams):  # unique per sentence
            bigram_sentence_counts[bg] += 1

    total_unique_bigrams = len(bigram_sentence_counts)
    if total_unique_bigrams == 0:
        return 0.0

    shared = sum(1 for c in bigram_sentence_counts.values() if c > 1)
    return shared / total_unique_bigrams


def _compute_pos_entropy(sentences: list[str], nlp) -> Optional[float]:
    """
    Compute Shannon entropy of the global POS tag distribution.

        H_POS = -sum(p_i * log2(p_i))  for each unique POS tag category

    Used internally as a subcomponent weight. NOT exposed as a separate feature.

    Args:
        sentences: List of sentence strings.
        nlp: spaCy Language model.

    Returns:
        float | None: POS entropy in bits, or None if no POS tags found.
    """
    all_pos: list[str] = []
    for sent in sentences:
        doc = nlp(sent)
        all_pos.extend(
            token.pos_ for token in doc
            if not token.is_space and not token.is_punct and token.pos_ not in {"", "X", "SPACE"}
        )
    if not all_pos:
        return None

    counts = Counter(all_pos)
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _compute_discourse_density(text: str) -> float:
    """
    Compute discourse-marker density as markers per 100 tokens.

        D = (total discourse-marker matches / total whitespace tokens) × 100

    Args:
        text: Full input text.

    Returns:
        float: Discourse density in [0.0, 100.0].
    """
    tokens = text.split()
    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0.0

    total_matches = sum(len(pat.findall(text)) for pat in _DISCOURSE_PATTERNS)
    return (total_matches / total_tokens) * 100.0


def get_structural_regularity(text: str) -> Dict[str, Optional[float]]:
    """
    Compute the Sentence & Structural Regularity composite feature.

    Subcomponents (all computed internally):
        A. starter_diversity      — float in [0.0, 1.0]  (higher = more diverse)
        B. pos_similarity         — float in [0.0, 1.0]  (higher = more uniform structure)
        C. pos_entropy            — float in bits         (internal weight)
        D. discourse_density      — float per 100 tokens  (higher = more markers)

    Composite formula:
        We invert starter_diversity so that ALL subcomponents point in the same
        direction: higher composite → more structurally uniform → more AI-like
        (subject to what the LR model actually learns).

        raw_composite = (
            (1 - starter_diversity)   * 0.35   +   # low diversity → AI-like
            pos_similarity            * 0.45   +   # high similarity → AI-like
            clamp(discourse_density / 10.0, 0, 1) * 0.20  # scaled to [0,1]
        )

        Note: These sub-weights are internal only. The Logistic Regression learns
        the final weight of the entire structural_regularity feature.

        composite_score = raw_composite × 100    →  [0.0, 100.0]

    If spaCy is unavailable:
        Uses only starter_diversity + discourse_density (no POS components).

    Args:
        text: Input text string.

    Returns:
        dict with keys:
            "starter_diversity"   (float | None)
            "pos_similarity"      (float | None)
            "pos_entropy"         (float | None)
            "discourse_density"   (float)
            "composite"           (float | None)  — in [0.0, 100.0]
            "spacy_available"     (bool)
            "n_sentences"         (int)
    """
    fallback = {
        "starter_diversity": None,
        "pos_similarity": None,
        "pos_entropy": None,
        "discourse_density": 0.0,
        "composite": None,
        "spacy_available": False,
        "n_sentences": 0,
    }

    if not text or not text.strip():
        logger.warning("get_structural_regularity: received empty text. Returning all-None.")
        return fallback

    sentences = _sentence_split(text)
    n_sents = len(sentences)

    if n_sents < _MIN_SENTENCES:
        logger.info(
            "get_structural_regularity: only %d sentences (< %d minimum). "
            "Returning None composite — structural signal unreliable.",
            n_sents,
            _MIN_SENTENCES,
        )
        d_density = _compute_discourse_density(text)
        return {
            **fallback,
            "discourse_density": round(d_density, 4),
            "n_sentences": n_sents,
        }

    # ---- A. Sentence starter diversity ----
    starter_div = _compute_starter_diversity(sentences)

    # ---- D. Discourse-marker density ----
    d_density = _compute_discourse_density(text)

    # ---- B + C. POS-based components (requires spaCy) ----
    nlp = _get_spacy()
    spacy_ok = nlp is not None
    pos_sim: Optional[float] = None
    pos_ent: Optional[float] = None

    if spacy_ok:
        try:
            pos_sim = _compute_pos_similarity(sentences, nlp)
            pos_ent = _compute_pos_entropy(sentences, nlp)
        except Exception as e:
            logger.warning("get_structural_regularity: POS computation error: %s", e)
            pos_sim = None
            pos_ent = None

    # ---- Composite ----
    # discourse_density scaled to [0, 1]: clamp at 10 per 100 tokens
    d_scaled = min(d_density / 10.0, 1.0)

    if spacy_ok and pos_sim is not None:
        raw = (
            (1.0 - starter_div) * 0.35
            + pos_sim * 0.45
            + d_scaled * 0.20
        )
    else:
        # Fallback without POS: redistribute weights
        raw = (
            (1.0 - starter_div) * 0.60
            + d_scaled * 0.40
        )

    composite = round(min(100.0, max(0.0, raw * 100.0)), 4)

    logger.debug(
        "get_structural_regularity: n_sents=%d  starter_div=%.4f  pos_sim=%s  "
        "pos_ent=%s  discourse_density=%.4f  composite=%.2f",
        n_sents,
        starter_div,
        f"{pos_sim:.4f}" if pos_sim is not None else "N/A",
        f"{pos_ent:.4f}" if pos_ent is not None else "N/A",
        d_density,
        composite,
    )

    return {
        "starter_diversity": round(starter_div, 6),
        "pos_similarity": round(pos_sim, 6) if pos_sim is not None else None,
        "pos_entropy": round(pos_ent, 6) if pos_ent is not None else None,
        "discourse_density": round(d_density, 4),
        "composite": composite,
        "spacy_available": spacy_ok,
        "n_sentences": n_sents,
    }
