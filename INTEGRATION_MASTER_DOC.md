# AI Forensics — Integration Master Documentation

**Status**: Integration Phase (Text + Audio)  
**Branch**: `Integration_branch`  
**Section 3C (Cross-Modal Check)**: Explicitly deferred — pending calibration data  
**Last updated**: 2026-08-14

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Text Forensics Pipeline](#2-text-forensics-pipeline)
3. [Audio Forensics Pipeline](#3-audio-forensics-pipeline)
4. [Fusion Integration Layer](#4-fusion-integration-layer)
5. [Unified Streamlit App](#5-unified-streamlit-app)
6. [Section 3C — Cross-Modal Check (Deferred)](#6-section-3c--cross-modal-check-deferred)
7. [Audio Badge Thresholds — Uncalibrated Caveat](#7-audio-badge-thresholds--uncalibrated-caveat)
8. [Repository File Map](#8-repository-file-map)
9. [How to Run](#9-how-to-run)
10. [Comprehensive Q&A Reference](#10-comprehensive-qa-reference)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      app.py (root)                              │
│              Unified Streamlit Interface                        │
│        Mode: Text Only | Audio Only | Combined                  │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐    ┌─────────────────────────────────────┐
│  fusion_integration  │    │     fusion_integration.py (root)    │
│  .run_full_pipeline()│    │  • Routes to text/audio pipelines   │
│                      │◄───│  • Assigns per-modality badges      │
│                      │    │  • Computes placeholder unified score│
│                      │    │  • Section 3C: deferred (raw delta) │
└──────┬───────────────┘    └─────────────────────────────────────┘
       │
       ├──────────────────────────────────────────────────────────►
       │                                                          │
       ▼                                                          ▼
┌─────────────────────────────────┐    ┌──────────────────────────────────┐
│  text_forensics/pipeline.py     │    │  audio_forensics/pipeline.py     │
│  analyze_text(text, robustness) │    │  analyze_audio(audio_path)       │
│                                 │    │                                  │
│  Signals:                       │    │  Pipeline:                       │
│   • Curvature (65%)             │    │   • strip_silence() — Silero VAD │
│   • Cliché Scan (5%)            │    │   • transcribe() — Whisper-Tiny  │
│   • Burstiness (20%)            │    │   • score_audio() — wav2vec2     │
│   • Lexical Entropy (10%)       │    │                                  │
│                                 │    │  Returns:                        │
│  Returns: text_score (0-100)    │    │   audio_score, logit_fake,       │
│   + signals, agreement, etc.    │    │   logit_real, transcript         │
└─────────────────────────────────┘    └──────────────────────────────────┘
```

---

## 2. Text Forensics Pipeline

### Entry Point
```python
from text_forensics.pipeline import analyze_text

result = analyze_text(text: str, run_robustness: bool = True) -> dict
```

### Locked Output Schema
```python
{
    "text_score": float,          # 0-100 fused AI-likelihood (higher = more AI-like)
    "signal_agreement": str,      # "agreement" | "disagreement"
    "signals": {
        "curvature_raw": float | None,
        "curvature_score": float | None,
        "burstiness_raw": float | None,
        "burstiness_score": float | None,
        "cliche_density_pct": float,
        "cliche_score": float,
        "ttr": float,
        "entropy": float,
        "entropy_score": float,
    },
    "stability_flag": str,         # "stable" | "unstable" | "skipped"
    "paraphrase_delta": float,
    "compared_on_truncated": bool,
}
```

### Signal Details

| Signal | Weight | Method | Raw Value | Calibration (Correction 15 Genre-Diverse) |
|--------|--------|--------|-----------|---------------------------------------------|
| **Probability Curvature** | **65%** | Fast-DetectGPT: log-likelihood discrepancy under `distilgpt2` | curvature_raw (lower = more AI-like) | Gaussian CDF: μ₀=−1.2804, σ₀=0.3973 (n=474) |
| **Burstiness** | **20%** | σ/μ of sentence lengths (NLTK tokenisation) | burstiness_raw | CDF, inverted (low = AI-like): μ₀=0.6561, σ₀=0.4112 (n=350) |
| **Lexical Entropy** | **10%** | Shannon entropy H of unigram distribution | entropy in bits | CDF, inverted (high = human): μ₀=6.1442, σ₀=0.7641 (n=475) |
| **Cliché Scan** | **5%** | Regex match on 50 AI buzzword terms | cliche_density_pct (% of total words) | Gaussian CDF on density: μ₀=0.0091, σ₀=0.0827 (n=475) |

### Fusion Formula
```
text_score = Σ (normalised_weight_i × CDF_score_i)
```
- Weights: `{curvature: 0.65, burstiness: 0.20, entropy: 0.10, cliche: 0.05}`
- Calibrated via **Correction 15** threshold-aware F1 grid-search: **ROC-AUC 0.9872, F1 0.9455** on 241-sample genre-diverse held-out set.
- Weights normalise dynamically if a signal returns `None` (e.g. burstiness on < 5 sentences).

### Text Verdict Thresholds (4-Way Classification System)
| Score Range | Verdict Badge | Description | Visual Style |
|-------------|---------------|-------------|--------------|
| **< 50.0** | 👤 **Human** | Confident human writing (core distribution) | Vibrant Green Card (`#22C55E`) |
| **50.0 – 69.9** | 👤 **Likely Human** | Leaning human (formal prose, journalistic writing) | Teal / Mint Card (`#14B8A6`) |
| **70.0 – 84.9** | 🤖 **Likely AI** | Leaning AI (mixed signals, partial AI editing) | Warm Amber Card (`#F97316`) |
| **≥ 85.0** | 🤖 **AI** | Confident AI generation (high curvature/clichés) | Solid Red Card (`#EF4444`) |

- **Method**: The 0–100 forensic score range is divided into 4 intuitive tiers. Mixed signals are split across `Likely Human` (50–70) and `Likely AI` (70–85) around the calibrated human/AI boundary ($t_{\text{human}}=69.74$).
- Saved in `text_forensics/calibration/fusion_config.json`.

### Calibration History

#### Correction 9 (initial)
- Grid search over weight combinations; curvature floor ≥ 0.40.
- Ranked by ROC-AUC, then flat-50 F1.
- Winner: `{curvature: 0.75, cliche: 0.20, burstiness: 0.05, entropy: 0.00}` — ROC-AUC 0.9994.
- Thresholds: hardcoded placeholders `human_max=45.0, ai_min=65.0` (never derived from data).

#### Correction 12 (threshold fix)
- Extracted real percentile statistics from the 120-sample tuning run.
- Replaced hardcoded thresholds with `t_human = h_90 = 76.52`, `t_ai = ai_10 = 86.50`, band width = 9.98 pts.

#### Correction 13 (threshold-aware F1 re-tune)
- Re-evaluated candidates against their own data-derived `t_ai` cutoff rather than a flat-50 ruler.
- Curvature floor lowered to 0.20. Winner: `{curvature: 0.50, cliche: 0.25, burstiness: 0.15, entropy: 0.10}`, thresholds `66.13 / 73.84`.

#### Correction 14 (manual entropy cap)
- Entropy capped at 0.03 to mitigate technical text genre-mismatch.
- Evaluation on formal legal text (Bar Council) revealed curvature (not entropy) was the primary driver of the genre mismatch on short formal prose.

#### Correction 15 (genre-diverse recalibration) — current
- **Corpus Composition (475 human calibration samples + 241 genre-diverse held-out samples)**:
  - **Conversational**: 250 HC3 `human_answers` (untouched)
  - **News**: 75 calibration + 25 holdout from `abisee/cnn_dailymail` (150–400 words)
  - **Legal/Bureaucratic**: 75 calibration + 25 holdout from `FiscalNote/billsum` (150–400 words)
  - **Technical**: 75 calibration + 25 holdout from `Salesforce/wikitext` wikitext-103 (150–400 words)
  - **AI Corpus**: 225 calibration + 66 holdout from `yahma/alpaca-cleaned` (LLaMA instruction outputs across news, legal, and technical registers).
- **Curvature Context Window & Token Ceiling**:
  - `signals/curvature.py` updated with a safety token ceiling of `max_tokens=512` (~350–400 words), preventing sequence overflow beyond `distilgpt2`'s 1024 token embedding capacity.
  - **Speed Benchmark** (CPU-only):
    - Short passage (~35 words): 0.07s cached
    - Long passage (~219 words): 0.22s cached (~3.1x longer, scaling sub-quadratically with token length)
- **Recomputed Baseline Stats (`baseline_stats.json`)**:
  - Curvature: $\mu_0 = -1.2804, \sigma_0 = 0.3973$ ($n=474$)
  - Burstiness: $\mu_0 = 0.6561, \sigma_0 = 0.4112$ ($n=350$)
  - Cliché: $\mu_0 = 0.0091, \sigma_0 = 0.0827$ ($n=475$)
  - Entropy: $\mu_0 = 6.1442, \sigma_0 = 0.7641$ ($n=475$)
- **Genre-Diverse Grid Search Results**:
  - Top 5 candidates all achieved F1 0.9455 and ROC-AUC ≥ 0.9866 on the 241-sample held-out set.
  - **Winner**: `{curvature: 0.65, burstiness: 0.20, entropy: 0.10, cliche: 0.05}` | $t_{\text{human}}=69.74, t_{\text{ai}}=78.00$ (band width 8.25 pts).
- **Validation Against Known Failure Cases**:
  1. **Transformers Technical Wikipedia Article** (Correction 6 failure case):
     - Score: **57.17** → **Likely Human-Written (< 69.74)** ✅ **PASS** (sub-scores: curvature 54.5, burstiness 67.2, cliche 100.0, entropy 32.9).
  2. **Bar Council News Article** (Correction 14 failure case):
     - Score: **81.84** → **Likely AI-Generated (≥ 78.00)** ⚠️ **Analysis**: The text is 110 words across 4 sentences (< 5 sentence guard), causing burstiness to return `None`. Without burstiness, curvature absorbs 81.25% of the fusion weight. While curvature sub-score improved from 95.08 to 88.3 under the new baseline, the absence of sentence rhythm data on short formal prose leaves the score curvature-dominated.
  3. **Conversational Regression Check** (30 HC3 samples):
     - **29/30 (96.7%) correct** (14/15 human correct, 15/15 AI correct) ✅ **PASS**.
- **Scope Note & Limitations**: The pipeline is now calibrated across 4 major genres (conversational, news, legal/bureaucratic, and technical/informational) totaling ~716 samples across calibration and evaluation. While significantly more robust than the HC3-only foundation, writing styles outside these domains (e.g., social media posts, short marketing copy, creative fiction, poetry) remain outside the current calibration distribution.

### Sentence-Level Highlighting
Uses a **sliding 3-sentence context window** `[S_{i−1}, S_i, S_{i+1}]` to give Fast-DetectGPT sufficient tokens for reliable curvature scoring. Each sentence is colored:
- 🔴 **Red**: windowed curvature score ≥ 50 (AI-likely)
- 🟢 **Green**: windowed curvature score < 50 (human-likely)
- ⚪ **Gray**: `None` (< 6 word context even with window)

### Adversarial Robustness Check
Paraphrases input via `T5_Paraphrase_Paws` and re-scores. If `|original_score − paraphrase_score| > 15.0`, stability_flag = `"unstable"`. Text inputs > 300 words are truncated for a fair comparison (`compared_on_truncated = True`).

---

## 3. Audio Forensics Pipeline

### Entry Point
```python
from audio_forensics.pipeline import analyze_audio

result = analyze_audio(audio_path: str) -> dict
```

### Locked Output Schema
```python
{
    "audio_score":         float,  # 0.0–100.0 deepfake confidence (higher = more AI)
    "logit_fake":          float,  # raw pre-softmax logit for 'fake' class
    "logit_real":          float,  # raw pre-softmax logit for 'real' class
    "transcript":          str,    # spoken text from Whisper-Tiny ASR
    "wer_confidence_note": str,    # fixed internal caveat string
}
```

### Pipeline Stages

| Stage | Module | Model | Purpose |
|-------|--------|-------|---------|
| 1. VAD | `vad.py → strip_silence()` | Silero VAD | Remove silence; resample to 16kHz mono |
| 2. ASR | `asr.py → transcribe()` | OpenAI Whisper-Tiny | Extract spoken transcript |
| 3. Deepfake | `deepfake_model.py → score_audio()` | `garystafford/wav2vec2-deepfake-voice-detector` | Classify as real/fake |

### Deepfake Scoring Formula
```
S_Audio = Sigmoid(logit_fake − logit_real) × 100
```
- label indices resolved **programmatically** (never hardcoded) and persisted to `audio_forensics/calibration/model_config.json`.

### Audio Verdict Thresholds (⚠️ UNCALIBRATED PLACEHOLDER)
| Score Range | Verdict |
|-------------|---------|
| ≥ 75 | Likely AI / Deepfake |
| 25 – 74.9 | Inconclusive |
| < 25 | Likely Human / Real Voice |

**⚠️ These thresholds are illustrative values from the architecture document.** They have NOT been validated against ground-truth audio data. They will be replaced after a Correction-9-style calibration run using matched real/fake audio clips.

---

## 4. Fusion Integration Layer

### Entry Point
```python
from fusion_integration import run_full_pipeline

result = run_full_pipeline(
    text: str | None = None,
    audio_path: str | None = None,
    run_robustness: bool = False,
) -> dict
```

### Locked Output Schema
```python
{
    # Text modality
    "text_score":            float | None,
    "text_verdict":          str   | None,
    "text_signals":          dict  | None,
    "text_signal_agreement": str   | None,
    "text_stability_flag":   str   | None,

    # Audio modality
    "audio_score":           float | None,
    "audio_verdict":         str   | None,   # includes [UNCALIBRATED] marker
    "audio_verdict_note":    str,            # always present — full caveat
    "logit_fake":            float | None,
    "logit_real":            float | None,
    "transcript":            str,

    # Cross-modal (Section 3C deferred)
    "cross_modal_deferred":    bool,         # always True
    "transcript_text_delta":   float | None, # raw |text_score − transcript_score|

    # Unified
    "unified_score":         float | None,
    "unified_verdict":       str   | None,
    "unified_score_note":    str,
}
```

### Unified Score Formula (Placeholder)
```
unified_score = mean(text_score, audio_score)  # equal-weight average
```
Equal weights are used as a **placeholder** until audio thresholds are validated. After audio calibration, weights should be recalibrated similarly to how text weights were grid-searched in Correction 9.

---

## 5. Unified Streamlit App

### Run Command
```bash
# From repository root (Integration_branch)
streamlit run app.py
```

### Analysis Modes
| Mode | What Runs | What Displays |
|------|-----------|---------------|
| 📝 Text Only | `analyze_text()` | Text score, sentence highlights, signal sub-scores, cliché evidence |
| 🎙️ Audio Only | `analyze_audio()` | Audio score, logit margins, transcript |
| 🔀 Combined | Both pipelines | Unified score banner + both modality panels |

### UI Components
- **Unified Score Banner**: Dark card with large score + verdict icon (Combined mode only)
- **Section 3C Deferred Notice**: Info box explaining why cross-modal check is not yet a badge
- **Sentence Highlights**: QuillBot-style red/green/gray span highlights (text panel)
- **Signal Sub-Score Bar Chart**: 4-bar Streamlit chart (text panel)
- **Cliché Evidence Tab**: Full text with AI buzzwords highlighted in red
- **Logit Metrics**: 4-column metric row (audio panel)
- **Uncalibrated Badge Warning**: Orange warning under audio verdict badge
- **Robustness Expander**: T5 stability result (text panel, optional)

---

## 6. Section 3C — Cross-Modal Check (Deferred)

### What It Is
Section 3C compares the text forensics score of the audio transcript against the original supplied text to detect a genuine cross-modal mismatch — e.g., a human-written document paired with a deepfake audio narration.

### Why It's Deferred
The delta threshold that separates a genuine mismatch from normal signal noise must be calibrated on real data — analogous to how text fusion weights were calibrated in Correction 9. Without matched real/fake audio clips to run this calibration, any hardcoded threshold would be arbitrary.

### Current Implementation
- `transcript_text_delta` is computed and returned in the output schema.
- `cross_modal_deferred = True` always.
- No Modality Conflict badge is issued.
- The raw delta is displayed in the UI as a data point for future calibration.

### What Comes Next
1. Collect matched pairs: `(audio_clip, text_input)` with known ground truth (matching vs. mismatched).
2. Score both modalities on each pair, compute the delta distribution.
3. Determine the threshold that maximises F1/ROC-AUC on the matched/mismatched classification task.
4. Implement the badge once the threshold is validated.

---

## 7. Audio Badge Thresholds — Uncalibrated Caveat

| Item | Status |
|------|--------|
| Text thresholds (≥75 AI, 50-75 Uncertain, <50 Human) | ✅ **Calibrated** — Correction 9, ROC-AUC 0.9994 |
| Audio thresholds (≥75 AI, 25-75 Inconclusive, <25 Human) | ⚠️ **UNCALIBRATED PLACEHOLDER** |

Audio thresholds come from the architecture document illustrative values. They have the same epistemic status text thresholds had before Correction 9. They are labeled `[UNCALIBRATED]` everywhere in code and UI and will be replaced after a calibration run using ground-truth audio clips.

---

## 8. Repository File Map

```
AI-audio-Text-Detection/           ← Integration_branch root
├── app.py                         ← [NEW] Unified Streamlit entry point
├── fusion_integration.py          ← [NEW] Integration fusion layer
├── requirements.txt               ← [UPDATED] Merged all dependencies
├── __init__.py                    ← [NEW] Root package marker
├── INTEGRATION_MASTER_DOC.md      ← [NEW] This document
├── .gitignore
│
├── text_forensics/                ← Shruti's text detection module (test_shruti)
│   ├── app.py                     ← Standalone text-only Streamlit app
│   ├── pipeline.py                ← analyze_text() — locked contract
│   ├── fusion.py                  ← compute_text_score(), load_fusion_config()
│   ├── robustness_test.py         ← check_stability() (T5 paraphrase)
│   ├── signals/
│   │   ├── curvature.py           ← get_curvature() — Fast-DetectGPT via distilgpt2
│   │   ├── burstiness.py          ← get_burstiness() — σ/μ sentence lengths
│   │   ├── cliche_scanner.py      ← get_cliche_density(), CLICHE_TERMS
│   │   ├── lexical_entropy.py     ← get_lexical_stats() — TTR + Shannon H
│   │   └── sentence_scorer.py     ← score_sentences() — sliding context window
│   ├── calibration/
│   │   ├── baseline_stats.json    ← μ₀/σ₀ per signal (calibrated on HC3)
│   │   ├── fusion_config.json     ← Correction 9 weights + thresholds
│   │   └── *.py                   ← Calibration scripts
│   ├── TEXT_FORENSICS_MASTER_DOC.md  ← Text-only detailed documentation
│   └── tests/
│
└── audio_forensics/               ← Anoushka's audio detection module (test_anoushka)
    ├── app.py                     ← Standalone audio-only Streamlit app
    ├── pipeline.py                ← analyze_audio() — locked contract
    ├── vad.py                     ← strip_silence() — Silero VAD
    ├── asr.py                     ← transcribe() — Whisper-Tiny
    ├── deepfake_model.py          ← score_audio() — wav2vec2 deepfake detector
    ├── calibration/
    │   └── model_config.json      ← Programmatically resolved label indices
    ├── sample_clips/              ← Test WAV clips
    │   ├── speech_sample.wav
    │   ├── test_clip_with_silence.wav
    │   └── heavy_silence_clip.wav
    ├── EXPLAINER.md               ← Audio pipeline detailed documentation
    └── tests/
```

---

## 9. How to Run

### Unified App (Integration_branch — recommended)
```bash
# From the repository root
pip install -r requirements.txt

# Also download NLTK data (first run)
python -c "import nltk; nltk.download('punkt_tab')"

# Also download spaCy model (if cliche_scanner uses it)
# python -m spacy download en_core_web_sm

# Launch unified app
streamlit run app.py
```

### Standalone Text App
```bash
cd text_forensics
pip install -r requirements.txt
streamlit run app.py
```

### Standalone Audio App
```bash
# From repo root
pip install -r requirements.txt
streamlit run audio_forensics/app.py
```

### Python API
```python
import sys
sys.path.insert(0, r"path/to/AI-audio-Text-Detection")

from fusion_integration import run_full_pipeline

# Text only
result = run_full_pipeline(text="Your text here", run_robustness=False)
print(result["text_score"], result["text_verdict"])

# Audio only
result = run_full_pipeline(audio_path="audio_forensics/sample_clips/speech_sample.wav")
print(result["audio_score"], result["audio_verdict"])

# Combined
result = run_full_pipeline(
    text="Your text here",
    audio_path="audio_forensics/sample_clips/speech_sample.wav",
)
print(result["unified_score"], result["unified_verdict"])
print("Cross-modal delta (raw):", result["transcript_text_delta"])
```

---

## 10. Comprehensive Q&A Reference

### Q1: How does the text pipeline detect AI text?
**Answer**: It fuses 4 independent statistical signals. Probability curvature (75% weight) measures if word choices sit at predictable log-probability peaks under `distilgpt2` — AI models generate text that consistently hits these peaks. Cliché scan (20%) flags 50 overused AI buzzwords. Burstiness (5%) measures sentence-length uniformity (AI tends to write uniform-length sentences). Lexical entropy (0% weight, collected for analysis) measures vocabulary richness. Each raw value is mapped to a 0–100 sub-score via Gaussian CDF using calibrated μ₀/σ₀ from the HC3 dataset.

### Q2: Why does a technical human-written paragraph score "Uncertain"?
**Answer**: Technical writing uses structured, uniform sentence lengths — the burstiness signal interprets this as AI-like. Meanwhile, curvature and entropy correctly score the text as human. The signal disagreement banner appears when sub-score spread > 40 points, alerting you that detectors conflict. The overall score reflects the weighted average, which in this case sits in the 50–69 "Uncertain" range.

### Q3: Why are audio badge thresholds marked [UNCALIBRATED]?
**Answer**: The text pipeline's thresholds (≥75 AI, 50–75 Uncertain, <50 Human) were determined by a Correction 9 grid-search achieving ROC-AUC 0.9994 on 120 HC3 ground-truth samples. No equivalent calibration has been done for audio. The architecture doc's illustrative values (≥75 AI, <25 Human) are placeholders with the same epistemic status text thresholds had before Correction 9 — clearly labeled as unvalidated to prevent confusion.

### Q4: What is Section 3C and why is it deferred?
**Answer**: Section 3C (Cross-Modal Consistency Check) compares the text forensics score of the audio transcript against the original text input to detect mismatches — e.g., a human-written document narrated by a deepfake voice. The delta threshold separating genuine mismatch from normal signal noise must be calibrated on matched ground-truth clip pairs, similar to how text weights were calibrated in Correction 9. Without that data, any hardcoded threshold would be arbitrary. The raw delta is computed and exposed (`transcript_text_delta`) for future use.

### Q5: Why does the sliding context window fix sentence highlighting?
**Answer**: Fast-DetectGPT probability curvature requires ~20–30 tokens to compute a statistically meaningful discrepancy. Single sentences (9–15 words) lacked sufficient context, causing AI-generated short sentences to score as human-likely. The 3-sentence sliding window `[S_{i−1}, S_i, S_{i+1}]` supplies adequate token context while visually attributing the score to the target sentence. Verified: all 5 sentences of an 83-word AI paragraph now score 86–99/100 (all red) after windowing.

### Q6: What does the unified score mean?
**Answer**: The unified score is an equal-weight average of `text_score` and `audio_score`. It is a **placeholder** for the final fusion formula, which should be calibrated once audio thresholds are validated. The `unified_score_note` field in the output schema always carries this caveat. Do not treat the equal-weight average as a validated, production-ready number.

### Q7: How does the VAD step work?
**Answer**: Silero VAD (a PyTorch-based voice activity detector) processes the audio at 16kHz and returns timestamps of speech segments. Silence segments are removed, and the remaining speech signal is concatenated. This improves ASR transcript quality and removes silence artifacts from the deepfake classifier input.

### Q8: How does the deepfake classifier work?
**Answer**: `garystafford/wav2vec2-deepfake-voice-detector` is a fine-tuned wav2vec2 model with two output logits (fake, real). The model's label mapping is resolved programmatically from `id2label` (never hardcoded) and persisted to `calibration/model_config.json`. The final score is `Sigmoid(logit_fake − logit_real) × 100`, mapping any logit difference to a 0–100% risk scale.

### Q9: How do I add more audio test clips for calibration?
**Answer**: Add `.wav` files to `audio_forensics/sample_clips/`. To run calibration for Section 3C, collect matched pairs: one audio clip of a human reading a text, and one deepfake audio reading the same text. Score both through `run_full_pipeline()`, record the `transcript_text_delta` values, and determine the threshold that maximises F1 on your ground-truth label (matched=0, mismatched=1).

### Q10: Why is the unified app at the repo root, not inside a sub-package?
**Answer**: The unified app imports from both `text_forensics.*` and `audio_forensics.*`. Placing it at the root eliminates relative import complexity and makes the run command `streamlit run app.py` (from root) clean and intuitive. The standalone per-modality apps remain inside their respective subdirectories for independent operation.
