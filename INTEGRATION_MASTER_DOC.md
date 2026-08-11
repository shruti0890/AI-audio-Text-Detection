# AI Forensics — Integration Master Documentation

**Status**: Integration Phase (Text + Audio)  
**Branch**: `Integration_branch`  
**Section 3C (Cross-Modal Check)**: Explicitly deferred — pending calibration data  
**Last updated**: 2026-08-11

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
│   • Curvature (75%)             │    │   • strip_silence() — Silero VAD │
│   • Cliché Scan (20%)          │    │   • transcribe() — Whisper-Tiny  │
│   • Burstiness (5%)             │    │   • score_audio() — wav2vec2     │
│   • Lexical Entropy (0%)        │    │                                  │
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

| Signal | Weight | Method | Raw Value | Calibration |
|--------|--------|--------|-----------|-------------|
| **Probability Curvature** | 75% | Fast-DetectGPT: log-likelihood discrepancy under `distilgpt2` | curvature_raw (lower = more AI-like) | Gaussian CDF: μ₀=−1.347, σ₀=0.326 |
| **Cliché Scan** | 20% | Regex match on 50 AI buzzword terms | cliche_density_pct (% of total words) | Gaussian CDF on density |
| **Burstiness** | 5% | σ/μ of sentence lengths (NLTK tokenisation) | burstiness_raw | CDF, inverted (low = AI-like) |
| **Lexical Entropy** | 0% | Shannon entropy H of unigram distribution | entropy in bits | CDF, inverted (high entropy = human-like) |

### Fusion Formula
```
text_score = Σ (normalised_weight_i × CDF_score_i)
```
- Weights: `{curvature: 0.75, cliche: 0.20, burstiness: 0.05, entropy: 0.00}`
- Calibrated via Correction 9 grid-search: **ROC-AUC 0.9994** on 120 HC3 held-out samples.

### Text Verdict Thresholds (Calibrated — Correction 9)
| Score Range | Verdict |
|-------------|---------|
| ≥ 75 | Likely AI-Generated |
| 50 – 74.9 | Uncertain / Mixed Signals |
| < 50 | Likely Human-Written |

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
