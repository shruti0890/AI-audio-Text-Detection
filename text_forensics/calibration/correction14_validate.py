"""
Correction 14 validation script.

1. Tests the news article text ("Manan Kumar Mishra / Bar Council of India")
   through the pipeline with the new weights and reports sub-scores + verdict.
2. Re-runs the full 120-sample HC3 set and reports ROC-AUC/F1 vs. Correction 13.

Run from project root:
    python text_forensics/calibration/correction14_validate.py
"""
from __future__ import annotations
import json
import logging
import sys
import numpy as np
from pathlib import Path

_CALIB_DIR = Path(__file__).resolve().parent
_PROJ_ROOT  = _CALIB_DIR.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
config = json.loads((_CALIB_DIR / "fusion_config.json").read_text(encoding="utf-8"))
W = config["weights"]
T_HUMAN = config["thresholds"]["human_max"]
T_AI    = config["thresholds"]["ai_min"]

logger.info("Loaded Correction 14 weights: %s", W)
logger.info("Thresholds: t_human=%.4f, t_ai=%.4f", T_HUMAN, T_AI)

# ---------------------------------------------------------------------------
# Part 1 — News article test case
# ---------------------------------------------------------------------------
NEWS_ARTICLE = (
    "Manan Kumar Mishra, the Chairperson of the Bar Council of India, "
    "has raised serious concerns about the increasing influence of artificial "
    "intelligence in the legal profession. Speaking at a national legal "
    "conference in New Delhi, he emphasised that while technology can assist "
    "lawyers, it must never replace human judgment, ethical reasoning, and "
    "the fundamental duty of an advocate to their client. "
    "Mishra warned that unregulated AI tools in courtrooms could undermine "
    "the adversarial system and compromise the principles of natural justice. "
    "He called on the Bar Council to frame guidelines for the responsible use "
    "of AI in legal practice, ensuring that accountability remains with the "
    "licensed advocate, not the algorithm."
)

logger.info("\n=== Part 1: News Article Test ===")
logger.info("Input length: %d words", len(NEWS_ARTICLE.split()))

from text_forensics.pipeline import analyze_text, _load_baseline_stats
from text_forensics.fusion import _cdf_score
from text_forensics.signals.curvature import get_curvature
from text_forensics.signals.burstiness import get_burstiness
from text_forensics.signals.cliche_scanner import get_cliche_density
from text_forensics.signals.lexical_entropy import get_lexical_stats

baseline = _load_baseline_stats()

curv_raw   = get_curvature(NEWS_ARTICLE)
burst_raw  = get_burstiness(NEWS_ARTICLE)
cliche_raw = get_cliche_density(NEWS_ARTICLE)
lex        = get_lexical_stats(NEWS_ARTICLE)
ent_raw    = lex["entropy"]

_INVERTED = {"burstiness", "entropy"}

def calibrate(key, raw_val):
    if raw_val is None:
        return None
    b_key = "cliche_density" if key == "cliche" else key
    stats  = baseline.get(b_key, {})
    mu0, sig0 = stats.get("mu0", 0.0), stats.get("sigma0", 1.0)
    cdf = _cdf_score(raw_val, mu0, sig0)
    v   = (100.0 - cdf) if key in _INVERTED else cdf
    return round(max(0.0, min(100.0, v)), 2)

c_curv  = calibrate("curvature",  curv_raw)
c_burst = calibrate("burstiness", burst_raw)
c_clich = calibrate("cliche",     cliche_raw)
c_ent   = calibrate("entropy",    ent_raw)

logger.info("Raw curvature : %s  -> sub-score: %s", curv_raw,   c_curv)
logger.info("Raw burstiness: %s  -> sub-score: %s", burst_raw,  c_burst)
logger.info("Raw cliche    : %s  -> sub-score: %s", cliche_raw, c_clich)
logger.info("Raw entropy   : %s  -> sub-score: %s", ent_raw,    c_ent)

# Fuse manually with C14 weights
signals = {"curvature": c_curv, "burstiness": c_burst, "cliche": c_clich, "entropy": c_ent}
avail_w  = {k: W[k] for k in W if signals[k] is not None}
tot_w    = sum(avail_w.values())
fused    = sum((avail_w[k] / tot_w) * signals[k] for k in avail_w)

if fused >= T_AI:
    verdict = f"Likely AI-Generated (>= {T_AI:.2f})"
elif fused >= T_HUMAN:
    verdict = f"Uncertain / Mixed ({T_HUMAN:.2f} – {T_AI:.2f})"
else:
    verdict = f"Likely Human-Written (< {T_HUMAN:.2f})"

logger.info("Fused score (C14 weights): %.2f", fused)
logger.info("Verdict: %s", verdict)

# ---------------------------------------------------------------------------
# Part 2 — 120-sample HC3 re-evaluation
# ---------------------------------------------------------------------------
logger.info("\n=== Part 2: 120-Sample HC3 Re-Evaluation ===")

HUMAN_POOL = _CALIB_DIR / "hc3_test_pool_human.json"
AI_POOL    = _CALIB_DIR / "hc3_test_pool_ai.json"

with open(HUMAN_POOL, encoding="utf-8") as f:
    all_human = json.load(f)
with open(AI_POOL, encoding="utf-8") as f:
    all_ai = json.load(f)

# Use all samples (same 120 as Correction 12 used for comparison parity)
texts  = all_human[:60] + all_ai[:60]
y_true = np.array([0]*60 + [1]*60)

logger.info("Scoring 120 samples with C14 weights...")

scores_c14 = []
for i, text in enumerate(texts):
    if (i+1) % 20 == 0:
        logger.info("  [%d/120]", i+1)
    cr  = get_curvature(text)
    br  = get_burstiness(text)
    cl  = get_cliche_density(text)
    en  = get_lexical_stats(text)["entropy"]
    sigs = {"curvature": calibrate("curvature", cr),
            "burstiness": calibrate("burstiness", br),
            "cliche":     calibrate("cliche",     cl),
            "entropy":    calibrate("entropy",    en)}
    aw = {k: W[k] for k in W if sigs[k] is not None}
    tw = sum(aw.values())
    sc = sum((aw[k]/tw)*sigs[k] for k in aw) if tw > 0 else 50.0
    scores_c14.append(sc)

scores_c14 = np.array(scores_c14)

# ROC-AUC (Mann-Whitney)
pos, neg = scores_c14[y_true==1], scores_c14[y_true==0]
u = sum(np.sum(p > neg) + 0.5*np.sum(p == neg) for p in pos)
roc_auc = u / (len(pos) * len(neg))

# F1 at t_ai (binary: >= t_ai → AI)
y_pred = (scores_c14 >= T_AI).astype(int)
tp = int(np.sum((y_pred==1)&(y_true==1)))
fp = int(np.sum((y_pred==1)&(y_true==0)))
fn = int(np.sum((y_pred==0)&(y_true==1)))
prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

# 3-band classification
verdicts = np.where(scores_c14 < T_HUMAN, "human",
           np.where(scores_c14 <= T_AI, "mixed", "ai"))
h_v = verdicts[:60]
a_v = verdicts[60:]

logger.info("C14 weights on 120 samples:")
logger.info("  ROC-AUC : %.4f   (C13 was 1.0000)", roc_auc)
logger.info("  F1(t_ai): %.4f   (C13 was 0.9474)", f1)
logger.info("  Precision: %.4f | Recall: %.4f", prec, rec)
logger.info("  3-band — Human: %d | Mixed: %d | AI: %d",
            np.sum(verdicts=="human"), np.sum(verdicts=="mixed"), np.sum(verdicts=="ai"))
logger.info("  GT Human (60): %d Human, %d Mixed, %d AI",
            np.sum(h_v=="human"), np.sum(h_v=="mixed"), np.sum(h_v=="ai"))
logger.info("  GT AI    (60): %d AI,    %d Mixed, %d Human",
            np.sum(a_v=="ai"), np.sum(a_v=="mixed"), np.sum(a_v=="human"))
