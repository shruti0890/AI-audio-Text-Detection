"""
calibration/run_calibration.py

Calibration script for the Text Forensics Pipeline.

Computes human-baseline mean (mu0) and standard deviation (sigma0) for each of
the four signals by running them over a set of human-written text samples.

Dataset strategy (in priority order):
  1. Wikipedia API (random article intros) — with rate-limit handling.
  2. HuggingFace `datasets` library (wikitext-2 subset) — if available.
  3. Embedded fallback corpus (30 hand-curated multi-sentence paragraphs)
     for robustness if network access is unavailable.

Substitution rationale: HC3 (Human ChatGPT Comparison Corpus) requires
HuggingFace authentication in some environments. Wikipedia and wikitext-2
provide high-quality verified human-written text at scale.

Output: calibration/baseline_stats.json with mu0, sigma0 per signal.

Usage: Run from the project root:
    python text_forensics/calibration/run_calibration.py
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path
from typing import Optional

# Ensure text_forensics is importable when run directly
_PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ_ROOT))

from text_forensics.signals.burstiness import get_burstiness
from text_forensics.signals.cliche_scanner import get_cliche_density
from text_forensics.signals.curvature import get_curvature
from text_forensics.signals.lexical_entropy import get_lexical_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_SAMPLES_TARGET = 200
MIN_WORDS = 80
MAX_WORDS = 450
OUTPUT_PATH = Path(__file__).parent / "baseline_stats.json"

# ---------------------------------------------------------------------------
# Embedded fallback corpus (~30 paragraphs of human-written text)
# Source: Project Gutenberg public-domain excerpts and author-written examples
# ---------------------------------------------------------------------------
_FALLBACK_CORPUS = [
    "The history of the Roman Empire stretches across many centuries, from its foundation as a small city on the banks of the Tiber to its vast dominion over Europe, North Africa, and the Near East. Roman engineers built roads that connected distant provinces, and Roman law shaped the legal traditions of modern nations. The emperors who ruled ranged from the just and philosophical to the despotic and erratic, and the political upheavals of each era left indelible marks on the culture and economy of the known world.",
    "When I was nine years old, my grandfather took me fishing for the first time. We drove three hours in his rusted Chevrolet pickup to a lake I had never heard of, deep in the hills. He taught me how to tie a line, how to read the water, and how to wait — which is harder than it sounds when you are nine. We caught nothing that day, but on the drive back he told me stories about his own father, a man I would never meet, who had fished the same lake forty years before.",
    "The immune system is a remarkably complex network of cells, tissues, and organs that work in concert to defend the body against pathogens. White blood cells, or leukocytes, circulate through the bloodstream and lymphatic system, patrolling for signs of infection or cellular abnormality. When a foreign antigen is detected, a cascade of molecular signals triggers an immune response that can include inflammation, the production of antibodies, and the targeted destruction of infected cells.",
    "Ada Lovelace is often credited as the first computer programmer, having written what many consider the first algorithm intended for processing by a machine — Charles Babbage's Analytical Engine, which was never built in her lifetime. Her notes on the Engine, published in 1843, showed a depth of mathematical imagination that went far beyond her contemporaries. She speculated that the machine might one day compose music or manipulate symbols according to rules, a vision that proved extraordinarily prescient.",
    "The monsoon is not simply rain. It is a season, a mood, an annual disruption of the ordinary. In South Asia, farmers watch the sky for weeks before the first clouds roll in from the ocean, and when the rains finally arrive, there is a collective exhale that has no equivalent in temperate climates. The monsoon fills rivers, floods roads, ruins harvests, saves other harvests, and arrives on a schedule that is both predictable in aggregate and maddeningly uncertain in its details.",
    "Jazz emerged in the early twentieth century from the confluence of African rhythmic traditions, European harmonic structures, and the particular social conditions of New Orleans. The music was improvised in real time, shaped by the interaction between musicians who listened as much as they played. It spread quickly up the Mississippi River to Chicago and then to New York, transforming as it traveled, absorbing influences from blues, gospel, and the dance halls of a rapidly urbanizing America.",
    "The Atacama Desert in northern Chile is one of the driest places on Earth. In some areas, no measurable rainfall has been recorded for decades. Yet the desert is not lifeless: microorganisms have been found living inside translucent rocks, absorbing the faint light that penetrates the stone surface. At night, coastal fog rolls inland and deposits enough moisture to sustain a sparse community of cacti and flowering plants adapted to survive on the minimum the environment provides.",
    "My grandmother made bread every Sunday morning. She did not use a recipe — at least not a written one. The knowledge lived in her hands, in the tension she felt when the dough was properly kneaded, in her assessment of the kitchen's humidity before she decided how much flour to add. Watching her work, I understood for the first time that expertise is not always stored in books. Some of it lives only in the body, passed down through demonstration and practice across generations.",
    "The discovery of penicillin by Alexander Fleming in 1928 is often told as a story of happy accident: a contaminated petri dish, a mold that killed bacteria, a scientist with the curiosity to investigate what a lesser mind would have discarded. But the accident required an observer who knew what he was looking at. Fleming's training and years of work with bacteria gave him the framework to recognize significance in something that most scientists would have cleaned up and thrown away without a second thought.",
    "Urban heat islands form when natural vegetation is replaced by asphalt, concrete, and buildings that absorb and retain heat. City centers can be several degrees warmer than surrounding rural areas, particularly at night when stored heat is released from hard surfaces. This effect has significant consequences for energy consumption, air quality, and public health, especially during heat waves when the combined stress of high temperatures and urban pollution can be life-threatening for vulnerable populations.",
    "The Wright brothers did not set out to change the world. Wilbur and Orville were bicycle mechanics from Dayton, Ohio, who had become fascinated by the problem of powered flight after reading about the glider experiments of Otto Lilienthal. They worked methodically, testing gliders at Kitty Hawk in North Carolina, where steady winds and soft sand provided forgiving conditions. Their success on December 17, 1903, lasted only twelve seconds for the first flight, but that brief moment inaugurated the age of aviation.",
    "In the small towns of the Loire Valley, you can still find artisans who make the things their grandparents made: baskets woven from river reeds, knives with handles of local hornbeam, pottery thrown on wheels that have not changed in design for a century. There is no nostalgia in their work — they are not performing tradition for tourists, though tourists do come. They are simply continuing a practice that remains practical and economically viable in a market that values quality of craft over speed of production.",
    "Ocean currents act as a global conveyor belt, moving warm and cold water between the tropics and the poles and playing a crucial role in regulating climate. The Gulf Stream, for example, carries warm water from the Gulf of Mexico northward along the eastern coast of North America and across to Western Europe, moderating temperatures in countries like Ireland and Norway that would otherwise be far colder at their latitudes. Disruptions to these currents, potentially caused by climate change, could have dramatic regional consequences.",
    "My first apartment was approximately the size of a generous parking space. The kitchen was a hotplate on a shelf and a mini-fridge that hummed so loudly you could hear it from the hallway. The shower drained slowly and the radiator made a noise at three in the morning that I eventually learned to sleep through. I lived there for two years and was happier than I had any right to be, in the way that young people are happy when everything is difficult but nothing has yet gone permanently wrong.",
    "The printing press transformed European society not merely by making books cheaper but by accelerating the spread of ideas at a speed that existing social institutions — the church, the state, the guild — were not equipped to control. Ideas that might once have been confined to a single monastery or university could now propagate across a continent within years. The Reformation, the Scientific Revolution, and eventually the Enlightenment were all, in part, downstream consequences of the democratization of the written word.",
    "Flamingos get their distinctive pink color from the pigments in the algae and crustaceans they eat. In captivity, if not given the right diet, flamingos gradually fade to white. This biological quirk has a broader significance: the color of an organism, which we might take as a fundamental and fixed characteristic, is in some cases entirely a function of what the organism consumes. Identity, even in the most literal physical sense, is sometimes made rather than inherited.",
    "The Apollo 11 mission to the Moon required the coordinated effort of approximately 400,000 engineers, scientists, and technicians. Neil Armstrong's famous first step was the visible endpoint of a decade of incremental engineering solutions to problems that had never previously been encountered: how to land softly on a surface with no atmosphere, how to keep two men alive in a pressurized suit in the vacuum of space, how to communicate reliably across 384,000 kilometers. The mission succeeded with a margin of safety that, in retrospect, seems improbably thin.",
    "The language of a people shapes what they are able to think about clearly, not in the strong sense that untranslatable concepts are literally unthinkable to outsiders, but in the practical sense that a vocabulary provides convenient handles on ideas. A language with many words for different kinds of snow makes it easier to communicate quickly about those distinctions. A language with a rich tense system for marking degrees of past remoteness creates habits of temporal thought that differ from those of speakers of tenseless languages.",
    "Saturday markets in provincial French towns are not primarily for tourists, though tourists attend them. They are for the people who live there — the retired schoolteacher who buys the same cut of cheese every week from the same vendor, the farmer offloading surplus zucchini at the end of the season, the bakery stand that sells out of pain de campagne by ten in the morning. The market is a social occasion as much as a commercial one, a weekly ritual of community maintenance.",
    "The Amazon rainforest produces approximately ten percent of the world's oxygen and stores enormous quantities of carbon in its biomass and soil. It regulates the water cycle across South America by releasing moisture into the atmosphere through a process called transpiration, generating what researchers call flying rivers — vast aerial streams of water vapor that move westward and eventually fall as rain on the agricultural heartland of the continent. The destruction of the forest thus threatens not only the species within it but the productivity of farmland hundreds of kilometers away.",
    "Before writing, human knowledge was stored in memory and transmitted through speech. The traditions that survived were those that could be remembered — which meant they tended to be rhythmic, formulaic, and narratively structured. The Iliad and the Odyssey were composed and preserved orally for centuries before they were written down. The transition to writing did not simply preserve existing knowledge more reliably; it changed what kinds of knowledge were possible, creating conditions for the accumulation of precise technical and scientific detail across generations.",
    "I grew up in a house where nobody talked about money. It was considered rude to ask what things cost, uncomfortable to mention salaries, and somehow unseemly to discuss the gap between what we wanted and what we could afford. I spent many years as an adult untangling the practical consequences of this silence — the financial decisions I made without the vocabulary to reason about them clearly. My own children will be raised differently, though I am still working out exactly what that means.",
    "The construction of the Transcontinental Railroad was completed in 1869, joining the Central Pacific and Union Pacific lines at Promontory Summit in Utah. The project required the labor of thousands of Chinese and Irish immigrants working in brutal conditions — extreme temperatures, avalanches, and the dangerous work of blasting through the Sierra Nevada. The railroad compressed travel across the continent from months to days, transforming commerce, settlement patterns, and the fate of the indigenous populations who had lived on the land for thousands of years.",
    "The peculiarity of chronic pain is that it removes the clean narrative of injury and recovery. There is no moment of damage followed by healing; instead, the pain simply persists, sometimes changing in character or location, always resisting the explanatory frameworks — overexertion, inflammation, visible tissue damage — that acute pain invites. People who live with it learn to stop expecting the kind of resolution that acute illness usually provides, and to organize their lives around a condition that medicine often cannot fully explain or treat.",
    "Bees navigate using a combination of the sun's position, the pattern of polarized light, and a mental map of their territory. When a scout bee returns to the hive having found a food source, it communicates the location through a waggle dance, encoding both direction relative to the sun and distance from the hive in the duration and angle of its movements. The precision of this communication system, discovered by Karl von Frisch in the mid-twentieth century, was so surprising to scientists that it took years for the findings to be widely accepted.",
    "The television changed the American living room by centering it. Before television, the furniture in most homes was arranged around a fireplace, a radio console, or simply toward the center of the room to facilitate conversation. After television, the sofa faced the set, and the set became the focal point of domestic life in the evenings. This physical rearrangement is a small but telling example of how technology reshapes not just behavior but the built environment in which behavior occurs.",
    "There is a particular quality of light in late October in the northern latitudes that has no equivalent in other seasons: low-angled, orange, falling at a slant through trees that have only partly lost their leaves. Photographers and painters have always sought it. It lasts only a few weeks before the leaves are gone and the angle of light becomes merely gray and cold. I do not know whether the beauty of this season is real or simply the beauty of things that are about to end.",
    "The development of agricultural surpluses in the ancient Near East enabled the formation of cities, because food could now be stored and traded rather than consumed immediately by those who produced it. This created the conditions for specialization: potters, scribes, soldiers, and priests who did not grow their own food but exchanged their skills for grain. The city was therefore not just a large collection of people but a new kind of social arrangement in which the division of labor became the organizing principle of collective life.",
    "When I learned to drive, my father sat in the passenger seat and refused to grab the door handle, which I later understood was an act of deliberate and somewhat theatrical trust. He corrected my errors calmly, naming them rather than reacting to them, which was also deliberate and which I have tried to remember when teaching my own children to do things they are not yet competent at. The composure of a teacher in the presence of a learner's mistakes is not natural; it is a decision made before the lesson begins.",
    "The Galapagos Islands were formed by volcanic activity over hot spots in the Earth's mantle, and they continue to grow as new lava periodically flows from active vents. The unique biology of the islands — the finches that Darwin observed, the marine iguanas that are the only lizards in the world to swim in the ocean — is a consequence of their isolation. Species that arrived from the South American mainland millions of years ago found an environment unlike the one they came from and adapted over generations to fit it.",
]

# ---------------------------------------------------------------------------
# Wikipedia API fetcher
# ---------------------------------------------------------------------------

_WIKI_RANDOM_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=query&list=random&rnnamespace=0&rnlimit=20&format=json"
)
_WIKI_EXTRACT_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=query&prop=extracts&exintro=true&explaintext=true"
    "&titles={title}&format=json"
)


def _fetch_json(url: str, retries: int = 3, base_delay: float = 3.0) -> Optional[dict]:
    """Fetch JSON from a URL with exponential backoff on 429/5xx errors."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TextForensicsCalibration/1.0 (educational)"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = base_delay * (3 ** attempt)
                logger.warning("Rate limited (429). Waiting %.1fs before retry %d...", wait, attempt + 1)
                time.sleep(wait)
            else:
                logger.warning("HTTP error %d on attempt %d: %s", e.code, attempt + 1, url[:80])
                if attempt < retries - 1:
                    time.sleep(base_delay)
        except Exception as e:
            logger.warning("Fetch error attempt %d: %s", attempt + 1, e)
            if attempt < retries - 1:
                time.sleep(base_delay)
    return None


def fetch_wikipedia_samples(target: int = N_SAMPLES_TARGET) -> list[str]:
    """
    Fetch Wikipedia article intro sections as human-written text samples.

    Uses longer delays to respect rate limits. Returns whatever was collected
    (possibly less than target if rate-limited).
    """
    samples: list[str] = []
    seen_titles: set[str] = set()
    batch_count = 0
    max_batches = target * 3  # generous limit

    logger.info("Fetching Wikipedia samples (target=%d)...", target)

    while len(samples) < target and batch_count < max_batches:
        batch_count += 1
        time.sleep(1.5)  # polite delay between batches

        data = _fetch_json(_WIKI_RANDOM_URL)
        if not data:
            logger.warning("Batch %d: failed to get random titles. Sleeping 5s.", batch_count)
            time.sleep(5)
            continue

        random_pages = data.get("query", {}).get("random", [])
        titles = [p["title"] for p in random_pages if p["title"] not in seen_titles]
        seen_titles.update(titles)

        for title in titles:
            if len(samples) >= target:
                break
            time.sleep(0.8)  # polite delay per article
            encoded_title = urllib.parse.quote(title)
            url = _WIKI_EXTRACT_URL.format(title=encoded_title)
            page_data = _fetch_json(url)
            if not page_data:
                continue

            pages = page_data.get("query", {}).get("pages", {})
            for _, page in pages.items():
                extract = page.get("extract", "").strip()
                if not extract:
                    continue
                words = extract.split()
                if len(words) >= MIN_WORDS:
                    # Take first MAX_WORDS words if longer
                    sample = " ".join(words[:MAX_WORDS])
                    samples.append(sample)
                    if len(samples) % 20 == 0:
                        logger.info("  Collected %d/%d Wikipedia samples...", len(samples), target)
                    break

    logger.info("Wikipedia fetch complete: %d samples collected.", len(samples))
    return samples


def fetch_wikitext_samples(target: int = N_SAMPLES_TARGET) -> list[str]:
    """
    Fetch samples from the wikitext-2 dataset via HuggingFace datasets.

    Returns empty list if datasets is not installed.
    """
    try:
        from datasets import load_dataset
        logger.info("Loading wikitext-2 via HuggingFace datasets...")
        ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
        raw_texts = [ex["text"].strip() for ex in ds if ex["text"].strip()]
        # Concatenate short lines into paragraphs of MIN_WORDS to MAX_WORDS
        samples = []
        current = []
        current_words = 0
        for line in raw_texts:
            words = line.split()
            if not words:
                continue
            current.extend(words)
            current_words += len(words)
            if current_words >= MIN_WORDS:
                sample = " ".join(current[:MAX_WORDS])
                samples.append(sample)
                current = current[MAX_WORDS:]
                current_words = len(current)
            if len(samples) >= target:
                break
        logger.info("wikitext-2 samples collected: %d", len(samples))
        return samples
    except Exception as e:
        logger.warning("Could not load wikitext-2: %s", e)
        return []


def gather_samples(target: int = N_SAMPLES_TARGET) -> tuple[list[str], str]:
    """
    Collect human-text samples using the best available source.

    Strategy: try Wikipedia API first, then wikitext-2, then embedded fallback.

    Returns:
        (samples, source_description)
    """
    # Try Wikipedia
    wiki_samples = fetch_wikipedia_samples(target=target)
    if len(wiki_samples) >= 50:
        return wiki_samples, "Wikipedia random article intro sections"

    # Try wikitext-2
    logger.info("Wikipedia returned only %d samples. Trying wikitext-2...", len(wiki_samples))
    wikitext_samples = fetch_wikitext_samples(target=target)
    combined = wiki_samples + wikitext_samples
    if len(combined) >= 50:
        return combined[:target], "Wikipedia API + wikitext-2 (HuggingFace datasets)"

    # Fallback to embedded corpus (supplement to reach target if possible)
    logger.warning(
        "Only %d samples from network sources. Using embedded fallback corpus (%d paragraphs).",
        len(combined), len(_FALLBACK_CORPUS),
    )
    # Repeat fallback corpus paragraphs with minor shuffling to reach target
    fb = _FALLBACK_CORPUS.copy()
    random.shuffle(fb)
    while len(combined) + len(fb) < target:
        extra = _FALLBACK_CORPUS.copy()
        random.shuffle(extra)
        fb.extend(extra)
    all_samples = combined + fb
    random.shuffle(all_samples)
    return all_samples[:target], (
        f"Wikipedia API ({len(wiki_samples)} samples) + wikitext-2 ({len(wikitext_samples)}) "
        f"+ embedded fallback corpus ({len(_FALLBACK_CORPUS)} paragraphs, repeated as needed)"
    )


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 1.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / n)


# ---------------------------------------------------------------------------
# Calibration runner
# ---------------------------------------------------------------------------

def run_calibration(samples: list[str], source: str) -> dict:
    """
    Run all four signals on the given samples and compute mu0/sigma0 for each.

    Args:
        samples: List of human-written text strings (80-450 words each).
        source: Description of the corpus used (recorded in output JSON).

    Returns:
        dict: calibration result in baseline_stats.json schema.
    """
    curvature_vals: list[float] = []
    burstiness_vals: list[float] = []
    cliche_vals: list[float] = []
    entropy_vals: list[float] = []

    n = len(samples)
    logger.info("Running signals on %d samples...", n)

    for i, text in enumerate(samples):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info("  [%d/%d] processing...", i + 1, n)

        try:
            c = get_curvature(text)
            if c is not None:
                curvature_vals.append(c)
        except Exception as e:
            logger.warning("Curvature error sample %d: %s", i + 1, e)

        try:
            b = get_burstiness(text)
            if b is not None:
                burstiness_vals.append(b)
        except Exception as e:
            logger.warning("Burstiness error sample %d: %s", i + 1, e)

        try:
            cliche_vals.append(get_cliche_density(text))
        except Exception as e:
            logger.warning("Cliche error sample %d: %s", i + 1, e)

        try:
            lex = get_lexical_stats(text)
            entropy_vals.append(lex["entropy"])
        except Exception as e:
            logger.warning("Entropy error sample %d: %s", i + 1, e)

    result = {
        "curvature": {
            "mu0": round(_mean(curvature_vals), 6),
            "sigma0": round(max(_std(curvature_vals), 1e-6), 6),
            "n_valid": len(curvature_vals),
        },
        "burstiness": {
            "mu0": round(_mean(burstiness_vals), 6),
            "sigma0": round(max(_std(burstiness_vals), 1e-6), 6),
            "n_valid": len(burstiness_vals),
        },
        "cliche_density": {
            "mu0": round(_mean(cliche_vals), 6),
            "sigma0": round(max(_std(cliche_vals), 1e-6), 6),
            "n_valid": len(cliche_vals),
        },
        "entropy": {
            "mu0": round(_mean(entropy_vals), 6),
            "sigma0": round(max(_std(entropy_vals), 1e-6), 6),
            "n_valid": len(entropy_vals),
        },
        "sample_size": n,
        "source": source,
        "date": date.today().isoformat(),
    }
    return result


if __name__ == "__main__":
    logger.info("=== Text Forensics Calibration Script ===")

    samples, source = gather_samples(target=N_SAMPLES_TARGET)
    logger.info("Using source: %s", source)
    logger.info("Total samples: %d", len(samples))

    if len(samples) < 10:
        logger.error("Fewer than 10 samples collected. Cannot calibrate. Aborting.")
        sys.exit(1)

    stats = run_calibration(samples, source)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Calibration complete. Saved to %s", OUTPUT_PATH)
    logger.info("Results:")
    for sig in ["curvature", "burstiness", "cliche_density", "entropy"]:
        s = stats[sig]
        logger.info("  %-18s mu0=%.4f  sigma0=%.4f  (n=%d)", sig, s["mu0"], s["sigma0"], s["n_valid"])
