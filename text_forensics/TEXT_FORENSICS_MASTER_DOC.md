# Text Forensics Deepfake Detector — Master Technical Implementation & Reference Documentation

---

## Executive Summary

The **Text Forensics Deepfake Detector** is a multi-signal ensemble framework designed to detect AI-generated text (ChatGPT, Claude, LLaMA, GPT-4) and distinguish it from human writing. 

Rather than relying on a single black-box classifier, the pipeline combines **4 independent statistical detectors**, a **sliding-window sentence-level explainability engine**, an **adversarial robustness self-test**, and an **ensemble fusion layer** with signal-disagreement detection.

---

## Table of Contents

1. [Architecture & System Design](#1-architecture--system-design)
2. [Core Signal Detectors (Statistical Breakdown)](#2-core-signal-detectors-statistical-breakdown)
   - 2.1 Probability Curvature (Fast-DetectGPT)
   - 2.2 Sentence Burstiness (Rhythm)
   - 2.3 AI Cliché & Buzzword Scanner
   - 2.4 Lexical Entropy & Type-Token Ratio
3. [Signal Fusion Engine & Disagreement Detection](#3-signal-fusion-engine--disagreement-detection)
4. [Sentence-Level Explainability Engine (Sliding Context Window)](#4-sentence-level-explainability-engine-sliding-context-window)
5. [Adversarial Robustness Check (T5 Paraphrasing)](#5-adversarial-robustness-check-t5-paraphrasing)
6. [Streamlit User Interface (`app.py`)](#6-streamlit-user-interface-apppy)
7. [Repository File Map & Dependencies](#7-repository-file-map--dependencies)
8. [Comprehensive Technical Q&A / Interview Reference](#8-comprehensive-technical-qa--interview-reference)

---

## 1. Architecture & System Design

The system follows a modular, decoupled pipeline architecture where signals operate independently and feed into a central calibration and fusion engine.

```
                            ┌─────────────────────────────────────────┐
                            │               Input Text                │
                            └────────────────────┬────────────────────┘
                                                 │
                  ┌──────────────────────────────┼──────────────────────────────┐
                  ▼                              ▼                              ▼
     ┌──────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
     │   Probability Curvature  │   │   Sentence Burstiness    │   │    AI Cliché Scanner     │
     │   (Fast-DetectGPT)       │   │   (Sentence Length σ/μ)  │   │    (50 Buzzwords)        │
     └────────────┬─────────────┘   └────────────┬─────────────┘   └────────────┬─────────────┘
                  │                              │                              │
                  └──────────────────────────────┼──────────────────────────────┘
                                                 │
                                                 ▼
                                    ┌──────────────────────────┐
                                    │     Lexical Entropy      │
                                    │  (Shannon Entropy & TTR) │
                                    └────────────┬─────────────┘
                                                 │
                                                 ▼
                                    ┌──────────────────────────┐
                                    │   Signal Fusion Engine   │
                                    │  (Weighted & Calibrated) │
                                    └────────────┬─────────────┘
                                                 │
                  ┌──────────────────────────────┴──────────────────────────────┐
                  ▼                                                             ▼
     ┌──────────────────────────┐                                  ┌──────────────────────────┐
     │  Sentence-Level Engine   │                                  │  Adversarial Robustness  │
     │  (Sliding 3-Sent Window) │                                  │  (T5 Paraphrase Delta)   │
     └────────────┬─────────────┘                                  └────────────┬─────────────┘
                  │                                                             │
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │
                                                 ▼
                                    ┌──────────────────────────┐
                                    │   Streamlit Web Interface│
                                    │   (Results & Evidence)   │
                                    └──────────────────────────┘
```

### Public API Contract Interface

The entire pipeline exposes a single public entry point in `text_forensics/pipeline.py`:

```python
analyze_text(text: str, run_robustness: bool = True) -> dict
```

#### Output Schema Contract:

```json
{
  "text_score": 81.9,
  "signal_agreement": "agreement", // "agreement" or "disagreement"
  "signals": {
    "curvature_raw": -0.8521,
    "curvature_score": 93.4,
    "burstiness_raw": 0.215,
    "burstiness_score": 88.4,
    "cliche_density_pct": 1.25,
    "cliche_score": 75.0,
    "ttr": 0.65,
    "entropy": 5.82,
    "entropy_score": 70.8
  },
  "stability_flag": "stable", // "stable", "unstable", or "skipped"
  "paraphrase_delta": 4.2,
  "compared_on_truncated": false
}
```

---

## 2. Core Signal Detectors (Statistical Breakdown)

### 2.1 Probability Curvature (`Fast-DetectGPT`)
- **Module**: `text_forensics/signals/curvature.py`
- **Core Concept**: LLMs generate text by selecting tokens near local probability log-likelihood peaks. Human writing exhibits higher variability in token probability space.
- **Methodology**: Calculates the log-likelihood discrepancy under `distilgpt2`:
  $$\tilde{d}(x) = \log p_\theta(x) - \mathbb{E}_{\tilde{x} \sim p_\theta(x)} [\log p_\theta(\tilde{x})]$$
- **Calibration Parameters**:
  - Baseline distribution parameters derived from HC3 dataset:
    $$\mu_0 = -1.346745, \quad \sigma_0 = 0.326020$$
  - Sub-score standard normal CDF conversion:
    $$z = \frac{\tilde{d} - \mu_0}{\sigma_0}, \quad \text{SubScore} = \text{round}\Big(100 \cdot \Phi(z)\Big)$$
- **Minimum Requirement**: Requires $\ge 20$ tokens to produce a valid score. Returns `None` if text is too short.

---

### 2.2 Sentence Burstiness (Rhythm)
- **Module**: `text_forensics/signals/burstiness.py`
- **Core Concept**: Human writers naturally vary sentence structures and lengths (short punchy sentences mixed with long complex ones). LLMs tend to generate sentences of uniform length.
- **Formula**: Sentence length variation coefficient:
  $$\text{Burstiness} = \frac{\sigma_{\text{sentence\_length}}}{\mu_{\text{sentence\_length}}}$$
- **Threshold**:
  - $\sigma/\mu \ge 0.35 \rightarrow$ High/Normal Burstiness (Human characteristic).
  - $\sigma/\mu < 0.35 \rightarrow$ Low Burstiness (AI characteristic).
- **Minimum Requirement**: Requires $\ge 5$ sentences to compute standard deviation vs mean. Returns `None` if $< 5$ sentences.

---

### 2.3 AI Cliché & Buzzword Scanner
- **Module**: `text_forensics/signals/cliche_scanner.py`
- **Core Concept**: Large language models exhibit strong stylistic biases toward specific transition words and corporate buzzwords (*"delve into"*, *"testament to"*, *"pivotal role"*, *"tapestry"*).
- **Implementation**: Case-insensitive regex boundary match scanning against a curated dictionary of 50 AI cliché phrases (`CLICHE_TERMS`).
- **Score Conversion**:
  $$\text{Cliché Density \%} = \left(\frac{\text{Total Cliché Words Matched}}{\text{Total Word Count}}\right) \times 100$$

---

### 2.4 Lexical Entropy & Type-Token Ratio
- **Module**: `text_forensics/signals/lexical_entropy.py`
- **Core Concept**: Measures vocabulary diversity (Type-Token Ratio $\text{TTR} = \frac{V}{N}$) and Shannon Entropy of unigram word frequencies:
  $$H(X) = -\sum_{i=1}^n P(x_i) \log_2 P(x_i)$$
- **Calibration**: Calibrated against the `v2_genre_mixed` dataset benchmark:
  $$\mu_0 = 7.12, \quad \sigma_0 = 0.45$$
  Lower entropy / repetitive vocabulary $\rightarrow$ higher AI sub-score.

---

## 3. Signal Fusion Engine & Disagreement Detection

- **Module**: `text_forensics/fusion.py`

### Fusion Weighting Schema
The overall text score $S_{\text{fused}} \in [0, 100]$ is computed as a weighted average of active (non-`None`) sub-scores:

$$\text{Weight Allocation} = \begin{cases}
\text{Curvature}: & 0.35 \\
\text{Burstiness}: & 0.25 \\
\text{Cliché Density}: & 0.20 \\
\text{Lexical Entropy}: & 0.20
\end{cases}$$

If a signal is missing (e.g. short text missing burstiness), its weight is dynamically redistributed proportionally among active signals.

### Signal Disagreement Detection Logic
If the maximum active sub-score minus the minimum active sub-score exceeds **40.0 points**:
$$\Delta_{\text{spread}} = \max(S_i) - \min(S_i) > 40.0$$
The pipeline tags `"signal_agreement": "disagreement"`. This triggers a red alert banner in the UI warning the user that individual detectors are providing conflicting evidence.

---

## 4. Sentence-Level Explainability Engine (Sliding Context Window)

- **Module**: `text_forensics/signals/sentence_scorer.py`

### Why Single-Sentence Isolation Failed
Scoring a 10–15 word sentence in total isolation did not provide Fast-DetectGPT enough tokens to calculate reliable probability log-likelihood curvature. As a result, individual sentences inside high-scoring AI paragraphs ($\ge 80/100$) were falsely highlighted as Green ("human-likely").

### The Sliding 3-Sentence Context Window Solution
To give the model sufficient tokens while attributing the result to the specific target sentence, `score_sentences()` constructs a 3-sentence sliding context window:

$$W_i = [S_{i-1}, S_i, S_{i+1}]$$

- **First Sentence ($S_0$)**: $W_0 = [S_0, S_1]$
- **Middle Sentences ($S_i$)**: $W_i = [S_{i-1}, S_i, S_{i+1}]$
- **Last Sentence ($S_{N-1}$)**: $W_{N-1} = [S_{N-2}, S_{N-1}]$
- **Short Texts ($\le 2$ sentences total)**: Evaluates sentence directly with `"low_context": True`.

### Binary Highlighting Thresholds
- 🔴 **Red (`.sent-high-ai`)**: Windowed curvature score $\ge 50$ (AI-likely).
- 🟢 **Green (`.sent-low-ai`)**: Windowed curvature score $< 50$ (Human-likely).
- ⚪ **Gray (`.sent-short`)**: `score is None` ($< 6$ words total even with window context).

---

## 5. Adversarial Robustness Check (T5 Paraphrasing)

- **Module**: `text_forensics/robustness_test.py`

### Methodology
To test whether a classification is fragile or dependent on superficial word choices:
1. Rephrases the input text using a `Vamsi/T5_Paraphrase_Paws` neural paraphrase model.
2. Re-scores the paraphrased text through the pipeline.
3. Computes the paraphrase delta: $\Delta_{\text{para}} = |S_{\text{original}} - S_{\text{paraphrased}}|$.

### Classification Rules
- $\Delta_{\text{para}} \le 15.0 \rightarrow$ `"stability_flag": "stable"`
- $\Delta_{\text{para}} > 15.0 \rightarrow$ `"stability_flag": "unstable"`
- Text $> 300$ words: Truncated to first 300 words for fair evaluation (`"compared_on_truncated": true`).
- UI Toggle: Controlled by `run_robustness` checkbox to avoid 30s delays when unneeded.

---

## 6. Streamlit User Interface (`app.py`)

The Streamlit UI ([app.py](file:///c:/Users/Hp/OneDrive/Desktop/AI-audio-Text-Detection/text_forensics/app.py)) provides a clean, focused user view:

1. **Input Area**: Radio toggle ("Paste Text" / "Upload .txt File"), word count stats, and `st.checkbox("Run robustness check (adds ~30s)", value=False)`.
2. **Analysis Results**:
   - Prominent Score Metric (`st.metric` with verdict label).
   - Signal Disagreement Alert (if triggered).
3. **Sentence-Level Breakdown**: Highlighted text block using binary Red/Green spans based on windowed curvature.
4. **Additional Explainability Evidence**:
   - Tab 1: Cliché word highlighting (`<mark class="cliche-highlight">`).
   - Tab 2: Rhythm & Burstiness table.
5. **Signal Sub-Scores Chart**: Interactive bar chart and sub-score table.
6. **Advanced Expanders**: Collapsed expanders for `Advanced: Robustness Check` and `Show Raw Signal Values (JSON)`.

---

## 7. Repository File Map & Dependencies

```
text_forensics/
├── app.py                      # Streamlit GUI Interface
├── pipeline.py                 # Main entry point: analyze_text()
├── fusion.py                   # Signal weighting & disagreement logic
├── robustness_test.py          # T5 paraphrase stability check
├── requirements.txt            # Python dependencies
├── TEXT_FORENSICS_MASTER_DOC.md # Master Documentation
├── signals/
│   ├── __init__.py
│   ├── curvature.py            # Fast-DetectGPT log-likelihood curvature
│   ├── burstiness.py           # Sentence length variation (σ/μ)
│   ├── cliche_scanner.py       # 50 AI buzzwords scanner
│   ├── lexical_entropy.py      # Shannon entropy & TTR
│   └── sentence_scorer.py      # Sliding 3-sentence window engine
├── calibration/
│   ├── baseline_stats.json     # Calibrated mu0/sigma0 parameters
│   └── run_calibration_hc3.py  # Calibration script for HC3 benchmark
└── tests/                      # Suite of unit & integration tests
```

---

## 8. Comprehensive Technical Q&A / Interview Reference

### Q1: Why does structured or technical human text sometimes get an "Uncertain / Mixed Signals (51.4)" score?
**Answer**: Technical, educational, or legal human writing often uses clean, uniform sentence structures. The **Burstiness detector** calculates sentence length variation ($\sigma/\mu$). If sentence lengths are uniform ($\sigma/\mu < 0.35$), burstiness flags a high AI sub-score (e.g. 88/100). Meanwhile, **Lexical Entropy** and **Cliché Scanner** recognize the human vocabulary and flag low scores (e.g. 26/100). The fusion engine detects this $> 40$-point spread and flags `"signal_agreement": "disagreement"`, cautioning the user that signals conflict.

### Q2: Why is sentence-level highlighting all GREEN when the overall score is 51.4?
**Answer**: Sentence-level breakdown relies **exclusively on Fast-DetectGPT probability curvature**, as burstiness and entropy require paragraph-length text to be statistically valid. If the author's word-choice predictability is human-like (~50), every sentence highlights GREEN. The overall score reached 51.4 because the paragraph-level **Burstiness** signal scored 88.4/100.

### Q3: Why did you switch from single-sentence isolated scoring to a sliding context window?
**Answer**: Fast-DetectGPT probability curvature requires sufficient token context ($\ge 20-30$ words) to calculate valid log-likelihood curvature. Single short sentences (9–12 words) lacked sufficient tokens, causing false negatives (green highlighting on AI sentences in an 80+ AI paragraph). The 3-sentence sliding window ($[S_{i-1}, S_i, S_{i+1}]$) supplies enough token context while visually attributing the score to the target sentence.

### Q4: How does the pipeline handle short inputs under 30 words?
**Answer**: Burstiness requires $\ge 5$ sentences and returns `None` for short text. The fusion engine dynamically redistributes burstiness's 0.25 weight proportionally across the remaining active signals (Curvature, Cliché, Entropy).

### Q5: How is adversarial robustness tested?
**Answer**: We run the input text through a `T5_Paraphrase_Paws` neural paraphrase model and re-score the paraphrased output. If the score shifts by $> 15.0$ points, the classification is flagged as `"unstable"`, warning that the verdict is sensitive to superficial phrasing.
