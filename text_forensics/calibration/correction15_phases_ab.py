"""
Correction 15 — Phase A + B: Genre-Diverse Corpus Collection.

Phase A: Collect ~100 human-written samples each from:
  - CNN/DailyMail  (news genre)
  - BillSum        (legal/bureaucratic genre)
  - Wikitext-103   (technical genre)

Phase B: Collect ~100 AI-written samples each from:
  - yahma/alpaca-cleaned (conversational-to-formal LLM outputs, news/technical/legal-ish)
    filtered for appropriate topic and length, tagged by genre.

Saves:
  - text_forensics/calibration/genre_corpus_additions_human.json
  - text_forensics/calibration/genre_corpus_additions_ai.json

Corpus split strategy (applied here to avoid Phase C/D overlap):
  - 75 samples per genre go to CALIBRATION set (Phase C baseline stats)
  - 25 samples per genre go to HELD-OUT set (Phase D grid search)
  - Tag field "split": "calibration" | "holdout"

Run from project root:
    python text_forensics/calibration/correction15_phases_ab.py
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_CALIB_DIR = Path(__file__).resolve().parent
_PROJ_ROOT  = _CALIB_DIR.parent.parent

# Per-genre targets
CALIB_PER_GENRE  = 75   # used in Phase C baseline
HOLDOUT_PER_GENRE = 25  # reserved for Phase D grid search
TOTAL_PER_GENRE  = CALIB_PER_GENRE + HOLDOUT_PER_GENRE  # 100

# Word-count window for samples
MIN_WORDS = 150
MAX_WORDS = 400


def _word_slice(text: str, min_w: int, max_w: int) -> str | None:
    """Return a min_w–max_w word slice of text, or None if too short."""
    words = text.split()
    if len(words) < min_w:
        return None
    return " ".join(words[:max_w])


def _tag(samples: list[str], genre: str, source: str, split_calib: int) -> list[dict]:
    """Attach genre/source/split metadata to raw text strings."""
    result = []
    for i, text in enumerate(samples):
        result.append({
            "text":   text,
            "genre":  genre,
            "source": source,
            "split":  "calibration" if i < split_calib else "holdout",
        })
    return result


# ---------------------------------------------------------------------------
# Phase A data sources
# ---------------------------------------------------------------------------

def fetch_cnn_dailymail(target: int) -> list[str]:
    """Fetch news articles from abisee/cnn_dailymail dataset."""
    from datasets import load_dataset
    logger.info("Loading abisee/cnn_dailymail (news)...")
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True)
    samples: list[str] = []
    for ex in ds:
        article = ex.get("article", "").strip()
        sliced = _word_slice(article, MIN_WORDS, MAX_WORDS)
        if sliced:
            samples.append(sliced)
        if len(samples) >= target:
            break
    logger.info("CNN/DailyMail: collected %d/%d samples.", len(samples), target)
    return samples


def fetch_billsum(target: int) -> list[str]:
    """Fetch US bill text from FiscalNote/billsum dataset (legal/bureaucratic genre)."""
    from datasets import load_dataset
    logger.info("Loading FiscalNote/billsum (legal)...")
    ds = load_dataset("FiscalNote/billsum", split="train", streaming=True)
    samples: list[str] = []
    for ex in ds:
        text = ex.get("text", "").strip()
        # Bill text often starts with section numbers / preamble — skip very short preamble
        # Remove section headers (lines starting with digits or "SEC.")
        lines = [l for l in text.splitlines() if not re.match(r"^\s*(SEC\.|SECTION|\d+\.)\s*$", l, re.I)]
        cleaned = " ".join(lines)
        sliced = _word_slice(cleaned, MIN_WORDS, MAX_WORDS)
        if sliced:
            samples.append(sliced)
        if len(samples) >= target:
            break
    logger.info("BillSum: collected %d/%d samples.", len(samples), target)
    return samples


def fetch_wikitext103(target: int) -> list[str]:
    """Fetch technical passages from wikitext-103-raw-v1."""
    from datasets import load_dataset
    logger.info("Loading wikitext-103-raw-v1 (technical)...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    samples: list[str] = []
    current_words: list[str] = []
    for ex in ds:
        line = ex.get("text", "").strip()
        if not line or line.startswith(" = "):
            # Section header or blank — flush current buffer if long enough
            if len(current_words) >= MIN_WORDS:
                sliced = " ".join(current_words[:MAX_WORDS])
                samples.append(sliced)
                if len(samples) >= target:
                    break
            current_words = []
            continue
        current_words.extend(line.split())
        if len(current_words) >= MAX_WORDS:
            sliced = " ".join(current_words[:MAX_WORDS])
            samples.append(sliced)
            current_words = current_words[MAX_WORDS:]
            if len(samples) >= target:
                break
    logger.info("Wikitext-103: collected %d/%d samples.", len(samples), target)
    return samples


# ---------------------------------------------------------------------------
# Phase B data sources
# ---------------------------------------------------------------------------

def fetch_alpaca_ai(target: int) -> list[str]:
    """
    Fetch AI-written text from yahma/alpaca-cleaned.
    Uses the 'output' field (LLaMA instruction-following outputs).
    Filters for outputs >= MIN_WORDS.
    """
    from datasets import load_dataset
    logger.info("Loading yahma/alpaca-cleaned (AI outputs)...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    samples: list[str] = []
    for ex in ds:
        output = ex.get("output", "").strip()
        sliced = _word_slice(output, MIN_WORDS, MAX_WORDS)
        if sliced:
            samples.append(sliced)
        if len(samples) >= target:
            break
    logger.info("Alpaca-cleaned AI outputs: collected %d/%d samples.", len(samples), target)
    return samples


def fetch_hc3_extended_ai(target: int, exclude_texts: set[str]) -> list[str]:
    """
    Fetch additional HC3 chatgpt_answers not already in the test pool.
    Used to supplement Alpaca for genre-diverse AI corpus.
    """
    from datasets import load_dataset
    logger.info("Loading HC3 for extended AI samples...")
    try:
        ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train", streaming=True)
    except Exception as e:
        logger.warning("Could not load HC3: %s", e)
        return []
    samples: list[str] = []
    for ex in ds:
        for ans in (ex.get("chatgpt_answers") or []):
            ans = ans.strip()
            if ans in exclude_texts:
                continue
            sliced = _word_slice(ans, MIN_WORDS, MAX_WORDS)
            if sliced and sliced not in exclude_texts:
                samples.append(sliced)
                exclude_texts.add(sliced)
            if len(samples) >= target:
                break
        if len(samples) >= target:
            break
    logger.info("HC3 extended AI: collected %d/%d samples.", len(samples), target)
    return samples


def assign_ai_genre(samples: list[str]) -> list[tuple[str, str]]:
    """
    Heuristically assign genre tags to AI samples by keyword matching.
    Returns list of (text, genre) tuples.
    """
    NEWS_KEYWORDS    = ["report", "announce", "government", "policy", "official",
                        "minister", "president", "parliament", "election", "bill",
                        "legislation", "court", "media", "journalist", "published"]
    LEGAL_KEYWORDS   = ["pursuant", "hereby", "shall", "section", "clause", "whereas",
                        "provision", "act", "statute", "regulation", "compliance",
                        "amendment", "plaintiff", "defendant", "jurisdiction"]
    TECHNICAL_KEYWORDS = ["algorithm", "function", "parameter", "process", "system",
                          "compute", "data", "analysis", "model", "equation",
                          "experiment", "hypothesis", "variable", "implement"]

    def score(text: str, kws: list[str]) -> int:
        t = text.lower()
        return sum(1 for k in kws if k in t)

    result = []
    genre_counts = {"news": 0, "legal": 0, "technical": 0}

    for text in samples:
        ns = score(text, NEWS_KEYWORDS)
        ls = score(text, LEGAL_KEYWORDS)
        ts = score(text, TECHNICAL_KEYWORDS)

        # Assign to lowest-count genre among top scorers (balance genres)
        ranked = sorted([("news", ns), ("legal", ls), ("technical", ts)],
                        key=lambda x: (-(x[1]), genre_counts[x[0]]))
        genre = ranked[0][0]
        genre_counts[genre] += 1
        result.append((text, genre))

    logger.info("AI genre assignment: %s", genre_counts)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    human_out = _CALIB_DIR / "genre_corpus_additions_human.json"
    ai_out    = _CALIB_DIR / "genre_corpus_additions_ai.json"

    # ---- Phase A: Human genre samples ----
    logger.info("\n=== PHASE A: Human Genre Samples ===")

    logger.info("--- News (CNN/DailyMail) ---")
    news_raw = fetch_cnn_dailymail(TOTAL_PER_GENRE)
    news_tagged = _tag(news_raw[:TOTAL_PER_GENRE], "news", "cnn_dailymail_3.0.0", CALIB_PER_GENRE)

    logger.info("--- Legal (BillSum) ---")
    legal_raw = fetch_billsum(TOTAL_PER_GENRE)
    legal_tagged = _tag(legal_raw[:TOTAL_PER_GENRE], "legal", "billsum", CALIB_PER_GENRE)

    logger.info("--- Technical (Wikitext-103) ---")
    tech_raw = fetch_wikitext103(TOTAL_PER_GENRE)
    tech_tagged = _tag(tech_raw[:TOTAL_PER_GENRE], "technical", "wikitext-103-raw-v1", CALIB_PER_GENRE)

    human_corpus = news_tagged + legal_tagged + tech_tagged
    with open(human_out, "w", encoding="utf-8") as f:
        json.dump(human_corpus, f, indent=2, ensure_ascii=False)
    logger.info("Phase A: saved %d human samples to %s", len(human_corpus), human_out)

    # Summary
    for genre in ["news", "legal", "technical"]:
        entries = [e for e in human_corpus if e["genre"] == genre]
        calib   = [e for e in entries if e["split"] == "calibration"]
        hold    = [e for e in entries if e["split"] == "holdout"]
        logger.info("  %-10s: %d total (%d calibration, %d holdout)", genre, len(entries), len(calib), len(hold))

    logger.info("\nPhase A example excerpts:")
    for genre in ["news", "legal", "technical"]:
        ex = [e for e in human_corpus if e["genre"] == genre]
        for i, sample in enumerate(ex[:2], 1):
            words = sample["text"].split()
            logger.info("[%s example %d] (%d words) %s...", genre, i, len(words), " ".join(words[:30]))

    # ---- Phase B: AI genre samples ----
    logger.info("\n=== PHASE B: AI Genre Samples ===")

    # Load existing AI test pool to avoid overlap
    ai_pool_path = _CALIB_DIR / "hc3_test_pool_ai.json"
    exclude_ai: set[str] = set()
    if ai_pool_path.exists():
        with open(ai_pool_path, encoding="utf-8") as f:
            exclude_ai = set(json.load(f))
    logger.info("Exclusion pool: %d existing AI test samples.", len(exclude_ai))

    # Try Alpaca first for AI samples
    ai_raw = fetch_alpaca_ai(TOTAL_PER_GENRE * 3)  # fetch more, then assign genres

    if len(ai_raw) < TOTAL_PER_GENRE:
        logger.warning("Alpaca returned only %d samples. Supplementing with HC3 extended.", len(ai_raw))
        ai_raw += fetch_hc3_extended_ai(TOTAL_PER_GENRE * 3 - len(ai_raw), exclude_ai)

    # Assign genres and select balanced subset
    ai_labelled = assign_ai_genre(ai_raw)

    # Balance: take TOTAL_PER_GENRE per genre
    ai_corpus: list[dict] = []
    for genre in ["news", "legal", "technical"]:
        subset = [(t, g) for t, g in ai_labelled if g == genre][:TOTAL_PER_GENRE]
        calib_n = min(CALIB_PER_GENRE, len(subset))
        for i, (text, g) in enumerate(subset):
            ai_corpus.append({
                "text":   text,
                "genre":  g,
                "source": "alpaca-cleaned (yahma/alpaca-cleaned) LLaMA outputs",
                "split":  "calibration" if i < calib_n else "holdout",
            })

    with open(ai_out, "w", encoding="utf-8") as f:
        json.dump(ai_corpus, f, indent=2, ensure_ascii=False)
    logger.info("Phase B: saved %d AI samples to %s", len(ai_corpus), ai_out)

    for genre in ["news", "legal", "technical"]:
        entries = [e for e in ai_corpus if e["genre"] == genre]
        calib   = [e for e in entries if e["split"] == "calibration"]
        hold    = [e for e in entries if e["split"] == "holdout"]
        logger.info("  %-10s: %d total (%d calibration, %d holdout)", genre, len(entries), len(calib), len(hold))

    logger.info("\nPhase B example excerpts:")
    for genre in ["news", "legal", "technical"]:
        ex = [e for e in ai_corpus if e["genre"] == genre]
        for i, sample in enumerate(ex[:2], 1):
            words = sample["text"].split()
            logger.info("[AI %s example %d] (%d words) %s...", genre, i, len(words), " ".join(words[:30]))

    logger.info("\n=== Phase A+B Complete ===")
    logger.info("Human corpus: %d total (%d calibration, %d holdout per genre target)",
                len(human_corpus),
                len([e for e in human_corpus if e["split"] == "calibration"]),
                len([e for e in human_corpus if e["split"] == "holdout"]))
    logger.info("AI corpus:    %d total (%d calibration, %d holdout per genre target)",
                len(ai_corpus),
                len([e for e in ai_corpus if e["split"] == "calibration"]),
                len([e for e in ai_corpus if e["split"] == "holdout"]))


if __name__ == "__main__":
    main()
