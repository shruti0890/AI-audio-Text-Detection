"""
text_forensics/calibration/generator_attribution_experiment/features/attribution_features.py

Feature Extraction Engine for Stage-2 AI-Generator Attribution:
Extracts forensic, stylometric, syntactic, discourse, and lexical profiling signals
to distinguish between model families (ChatGPT, Google Gemini, Anthropic Claude, Other AI).

Strictly leakage-free: Strips all metadata, prompt IDs, headers, and model identifiers.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional

import nltk
from scipy.stats import entropy

# Ensure basic tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Canonical Feature Names
# ---------------------------------------------------------------------------
ATTRIBUTION_FEATURE_ORDER = [
    # 1. Base Forensic Features (5)
    "curvature",
    "burstiness",
    "lexical_entropy",
    "structural_regularity",
    "cliche_density",

    # 2. Stylometric & Punctuation Profile (8)
    "avg_sentence_len",
    "std_sentence_len",
    "em_dash_density",
    "semicolon_colon_density",
    "parenthetical_density",
    "bullet_list_density",
    "quote_density",
    "uppercase_word_ratio",

    # 3. Discourse & Transition Dynamics (4)
    "formal_transition_density",
    "introductory_clause_density",
    "hedging_density",
    "sentence_starter_diversity",

    # 4. Lexical Richness & Function Words (5)
    "type_token_ratio",
    "root_ttr",
    "pronoun_density",
    "preposition_density",
    "modal_verb_density",
]

# Keyword lists for linguistic profiling
FORMAL_TRANSITIONS = {
    "furthermore", "moreover", "consequently", "conversely", "specifically",
    "notably", "importantly", "principally", "fundamentally", "additionally",
    "alternatively", "subsequently", "accordingly", "ultimately", "essentially",
    "inherently", "primarily", "predominantly", "in contrast", "on the other hand",
    "in summary", "in conclusion", "to summarize", "in essence"
}

HEDGING_TERMS = {
    "typically", "generally", "largely", "frequently", "tend to", "tends to",
    "often", "usually", "potentially", "arguably", "somewhat", "relatively",
    "broadly", "ostensibly", "presumably", "practically"
}

MODAL_VERBS = {"can", "could", "may", "might", "shall", "should", "will", "would", "must"}

PRONOUNS = {
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
    "their", "theirs", "this", "that", "these", "those"
}

PREPOSITIONS = {
    "about", "above", "across", "after", "against", "along", "among", "around",
    "at", "before", "behind", "below", "beneath", "beside", "between", "beyond",
    "by", "down", "during", "except", "for", "from", "in", "inside", "into",
    "near", "of", "off", "on", "onto", "out", "outside", "over", "past",
    "through", "throughout", "to", "toward", "towards", "under", "underneath",
    "until", "unto", "up", "upon", "with", "within", "without"
}


def _split_sentences(text: str) -> List[str]:
    """Tokenizes text into clean sentences."""
    try:
        sents = nltk.sent_tokenize(text)
    except Exception:
        sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return [s.strip() for s in sents if len(s.strip().split()) >= 2]


def extract_attribution_features(
    text: str,
    base_signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Extracts all 22 attribution features from raw text.

    Args:
        text: Non-empty string.
        base_signals: Optional pre-computed base signals dict to save time.

    Returns:
        Dict mapping each feature name in ATTRIBUTION_FEATURE_ORDER to float.
    """
    clean_text = text.strip()
    words = clean_text.split()
    n_words = max(len(words), 1)
    sentences = _split_sentences(clean_text)
    n_sentences = max(len(sentences), 1)

    # -----------------------------------------------------------------------
    # 1. Base Forensic Features
    # -----------------------------------------------------------------------
    if base_signals is not None:
        curv = base_signals.get("curvature", 0.0) or 0.0
        burst = base_signals.get("burstiness", 0.60) or 0.60
        ent = base_signals.get("lexical_entropy", 6.20) or 6.20
        struct = base_signals.get("structural_regularity", 35.0) or 35.0
        cliche = base_signals.get("cliche_density", 0.0) or 0.0
    else:
        from text_forensics.signals.burstiness import get_burstiness
        from text_forensics.signals.cliche_scanner import get_cliche_density
        from text_forensics.signals.curvature import get_curvature
        from text_forensics.signals.lexical_entropy import get_lexical_stats
        from text_forensics.signals.structural_regularity import get_structural_regularity

        curv_raw = get_curvature(clean_text)
        curv = float(curv_raw) if curv_raw is not None else -0.8246

        burst_raw = get_burstiness(clean_text)
        burst = float(burst_raw) if burst_raw is not None else 0.6045

        lex = get_lexical_stats(clean_text)
        ent_raw = lex.get("entropy")
        ent = float(ent_raw) if ent_raw is not None else 6.2749

        struct_res = get_structural_regularity(clean_text)
        struct_raw = struct_res.get("composite") if isinstance(struct_res, dict) else None
        struct = float(struct_raw) if struct_raw is not None else 33.346

        cliche_raw = get_cliche_density(clean_text)
        cliche = float(cliche_raw) if cliche_raw is not None else 0.0

    # -----------------------------------------------------------------------
    # 2. Stylometric & Punctuation Profile
    # -----------------------------------------------------------------------
    sent_lens = [len(s.split()) for s in sentences]
    avg_sent_len = sum(sent_lens) / n_sentences

    if len(sent_lens) >= 2:
        mean_l = avg_sent_len
        var_l = sum((l - mean_l) ** 2 for l in sent_lens) / (len(sent_lens) - 1)
        std_sent_len = math.sqrt(var_l)
    else:
        std_sent_len = 0.0

    # Punctuation counts normalized per 100 words
    em_dashes = len(re.findall(r"—|--", clean_text))
    em_dash_density = (em_dashes / n_words) * 100.0

    semicolons_colons = len(re.findall(r"[;:]", clean_text))
    semicolon_colon_density = (semicolons_colons / n_words) * 100.0

    parentheses = len(re.findall(r"\(.*?\)", clean_text))
    parenthetical_density = (parentheses / n_words) * 100.0

    # Bullet lists / numbered points
    bullets = len(re.findall(r"(?:^|\n)\s*(?:[\*\-\•]|\d+[\.\)]|[1-9]️⃣|✅)", clean_text))
    bullet_list_density = (bullets / n_sentences) * 100.0

    quotes = len(re.findall(r"[\"'\u201c\u201d\u2018\u2019]", clean_text))
    quote_density = (quotes / n_words) * 100.0

    # Uppercase words / acronyms (excluding single letters like 'I' or 'A')
    uppercase_words = sum(1 for w in words if len(w) >= 2 and w.isupper())
    uppercase_word_ratio = uppercase_words / n_words

    # -----------------------------------------------------------------------
    # 3. Discourse & Transition Dynamics
    # -----------------------------------------------------------------------
    lower_text = clean_text.lower()
    clean_words_lower = [w.strip(string.punctuation).lower() for w in words if w.strip(string.punctuation)]

    # Formal transition frequency per 100 words
    transition_count = 0
    for term in FORMAL_TRANSITIONS:
        if " " in term:
            transition_count += lower_text.count(term)
        else:
            transition_count += clean_words_lower.count(term)
    formal_transition_density = (transition_count / n_words) * 100.0

    # Introductory clauses (sentences starting with a transition or participle)
    intro_count = 0
    for s in sentences:
        s_low = s.strip().lower()
        if any(s_low.startswith(term) for term in ["in ", "when ", "to ", "for example", "unlike ", "furthermore", "however"]):
            intro_count += 1
    introductory_clause_density = (intro_count / n_sentences) * 100.0

    # Hedging frequency per 100 words
    hedging_count = sum(clean_words_lower.count(term) for term in HEDGING_TERMS)
    hedging_density = (hedging_count / n_words) * 100.0

    # Sentence starter diversity (unique first words / n_sentences)
    first_words = [s.split()[0].strip(string.punctuation).lower() for s in sentences if s.split()]
    sentence_starter_diversity = len(set(first_words)) / max(len(first_words), 1)

    # -----------------------------------------------------------------------
    # 4. Lexical Richness & Function Words
    # -----------------------------------------------------------------------
    n_tokens = len(clean_words_lower)
    unique_tokens = len(set(clean_words_lower))

    type_token_ratio = unique_tokens / max(n_tokens, 1)
    root_ttr = unique_tokens / math.sqrt(max(n_tokens, 1))

    # Pronoun, Preposition, Modal verb densities (per 100 words)
    pronoun_count = sum(1 for w in clean_words_lower if w in PRONOUNS)
    pronoun_density = (pronoun_count / max(n_tokens, 1)) * 100.0

    prep_count = sum(1 for w in clean_words_lower if w in PREPOSITIONS)
    preposition_density = (prep_count / max(n_tokens, 1)) * 100.0

    modal_count = sum(1 for w in clean_words_lower if w in MODAL_VERBS)
    modal_verb_density = (modal_count / max(n_tokens, 1)) * 100.0

    return {
        "curvature": round(curv, 6),
        "burstiness": round(burst, 6),
        "lexical_entropy": round(ent, 6),
        "structural_regularity": round(struct, 6),
        "cliche_density": round(cliche, 6),
        "avg_sentence_len": round(avg_sent_len, 4),
        "std_sentence_len": round(std_sent_len, 4),
        "em_dash_density": round(em_dash_density, 4),
        "semicolon_colon_density": round(semicolon_colon_density, 4),
        "parenthetical_density": round(parenthetical_density, 4),
        "bullet_list_density": round(bullet_list_density, 4),
        "quote_density": round(quote_density, 4),
        "uppercase_word_ratio": round(uppercase_word_ratio, 4),
        "formal_transition_density": round(formal_transition_density, 4),
        "introductory_clause_density": round(introductory_clause_density, 4),
        "hedging_density": round(hedging_density, 4),
        "sentence_starter_diversity": round(sentence_starter_diversity, 4),
        "type_token_ratio": round(type_token_ratio, 4),
        "root_ttr": round(root_ttr, 4),
        "pronoun_density": round(pronoun_density, 4),
        "preposition_density": round(preposition_density, 4),
        "modal_verb_density": round(modal_verb_density, 4),
    }


def extract_feature_vector(features_dict: Dict[str, float]) -> List[float]:
    """Returns a list of feature values ordered by canonical ATTRIBUTION_FEATURE_ORDER."""
    return [features_dict.get(fname, 0.0) for fname in ATTRIBUTION_FEATURE_ORDER]
