# 🔬 Multimodal AI Audio & Text Forensics System

> **A Unified Deepfake Speech & AI-Generated Text Detection Platform featuring an Ensemble Analysis Pipeline and Cross-Modality Consistency Reasoner.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/shruti0890/AI-audio-Text-Detection/pulls)

---

## 📌 Executive Summary

With the rapid progression of Generative AI, synthetic voice cloning (TTS/VC) and Large Language Model (LLM) text generation have enabled hyper-realistic digital spoofing. The **Multimodal AI Audio & Text Forensics System** is an end-to-end forensic platform engineered to detect synthetic speech, AI-generated text, and **cross-modal inconsistency attack vectors**.

Rather than relying on single black-box models, this system deploys:
1. **Audio Deepfake Forensics Engine**: Combines **Silero VAD** for silence removal, **Wav2Vec2** acoustic feature classification, and **Whisper ASR** for automatic speech-to-text transcription.
2. **Text Forensics Engine**: Multi-signal statistical ensemble combining **SmolLM2-135M** probability curvature (Fast-DetectGPT), sentence **burstiness** ($\sigma/\mu$), Shannon **lexical entropy**, **AI cliché** density scanning, and a calibrated **5-feature Logistic Regression classifier**.
3. **Cross-Modality Reasoner**: A deterministic Stage-2 decision matrix that aligns acoustic voice characteristics against spoken semantic text patterns to identify spoofing vectors like *human voice reading an LLM script* or *AI voice cloning human writing*.

---

## 📐 System Architecture

```
                                  ┌──────────────────────────┐
                                  │       USER INPUT         │
                                  └────────────┬─────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
          🎙️ Audio Input (File / Mic)                        📝 Text Input (Direct)
                       │                                               │
                       ▼                                               │
        ┌─────────────────────────────┐                                │
        │ Silero VAD (Silence Trim)   │                                │
        └──────────────┬──────────────┘                                │
                       │                                               │
           ┌───────────┴───────────┐                                   │
           ▼                       ▼                                   │
┌────────────────────┐   ┌───────────────────┐                         │
│ Wav2Vec2 Acoustic  │   │ Whisper-Tiny ASR  │                         │
│ Deepfake Model     │   │ Speech-to-Text    │                         │
└──────────┬─────────┘   └─────────┬─────────┘                         │
           │                       │                                   │
           │                       └─────────────────┐                 │
           ▼                                         ▼                 ▼
 ┌──────────────────┐                     ┌──────────────────────────────────┐
 │ Audio Score (0-1)│                     │      Text Forensics Engine       │
 │ 4-Tier Verdict   │                     │  • SmolLM2 Probability Curvature │
 └─────────┬────────┘                     │  • Sentence Length Burstiness    │
           │                              │  • Lexical Entropy & TTR         │
           │                              │  • AI Cliché & Buzzword Density  │
           │                              │  • 5-Feature Logistic Regression │
           │                              └────────────────┬─────────────────┘
           │                                               │
           │                                               ▼
           │                                     ┌──────────────────┐
           │                                     │ Text Score (0-100│
           │                                     │ 4-Tier Verdict   │
           │                                     └─────────┬────────┘
           │                                               │
           └───────────────────────┬───────────────────────┘
                                   ▼
                   ┌──────────────────────────────┐
                   │    Cross-Modality Reasoner   │
                   │  (Acoustic vs Text Alignment)│
                   └───────────────┬──────────────┘
                                   ▼
                   ┌──────────────────────────────┐
                   │     Unified Web UI           │
                   │   (Streamlit Dashboard)      │
                   └──────────────────────────────┘
```

---

## ✨ Key Features & Highlights

### 1. 🎙️ Audio Forensics Pipeline
- **Silero VAD Silence Trimming**: Strips dead air and low-energy segments to prevent silent gaps from skewing acoustic feature extraction.
- **Wav2Vec2 Deepfake Classifier**: Leverages fine-tuned self-supervised speech representations with sliding window analysis ($T=1.15$ temperature scaling) to detect TTS synthesis artifacts, vocoder phase anomalies, and neural voice cloning.
- **Whisper ASR Integration**: OpenAI Whisper-Tiny extracts verbatim text transcripts from input audio files or live microphone recordings automatically.
- **Live Microphone Stream**: Real-time browser audio capture via WebAudio MediaRecorder API.

### 2. 📝 Text Forensics Pipeline
- **SmolLM2-135M Probability Curvature**: Implements **Fast-DetectGPT** curvature measurement by computing log-likelihood discrepancies ($\mathbf{d}(x) = \log p(x) - \mathbb{E}[\log p(\tilde{x})]$) under local perturbations.
- **Sentence Burstiness ($\sigma/\mu$)**: Analyzes structural variance in sentence length. AI models generate uniform sentence lengths (low burstiness), whereas human writing displays dynamic rhythm.
- **Shannon Lexical Entropy**: Measures vocabulary richness and token distribution predictability ($H = -\sum p(x) \log_2 p(x)$).
- **AI Cliché Density Scanner**: Scans for 50+ overused LLM buzzwords (*delve, tapestry, paramount, testaments, pivotal, crucial, underscore*).
- **Calibrated 5-Feature Logistic Regression**: Fuses signals into a 0–100 probability score mapped to 4 confidence tiers (*Authentic Human, Likely Human, Likely AI, Authentic AI*).

### 3. ⚡ Cross-Modality Consistency Reasoner
Detects subtle spoofing vectors by comparing voice authenticity against text authenticity:

| State Classification | Audio Verdict | Text Verdict | Forensic Interpretation |
| :--- | :---: | :---: | :--- |
| **`HUMAN_HUMAN`** | Human Voice | Human Text | **Modality Consistent**: Genuine human voice speaking authentic human-written content. |
| **`AI_AI`** | AI Voice | AI Text | **Modality Consistent**: Fully synthetic deepfake video/audio generated by AI pipeline. |
| **`HUMAN_VOICE_AI_TEXT_CONFLICT`** | Human Voice | AI Text | ⚠️ **Modality Conflict**: Authentic human voice reading an LLM-generated script (e.g. ChatGPT speech). |
| **`AI_VOICE_HUMAN_TEXT_CONFLICT`** | AI Voice | Human Text | ⚠️ **Modality Conflict**: Synthetic voice clone reading authentic human writing (e.g. TTS voiceover). |

---

## 🗂️ Project Structure

```
AI-audio-Text-Detection/
├── app.py                         # Unified Streamlit Web Application (Root Entry Point)
├── fusion_integration.py          # Unified Multi-Modal Orchestrator & Pipeline Runner
├── cross_modal_reasoner.py        # Stage-2 Cross-Modality Decision Engine
├── requirements.txt               # Unified Python Dependencies
├── AUDIO_CALIBRATION_PROTOCOL.md  # Audio Threshold Calibration Specification
├── INTEGRATION_MASTER_DOC.md      # Unified System Master Documentation
│
├── audio_forensics/               # Audio Forensics Subsystem
│   ├── app.py                     # Standalone Audio Forensics Streamlit App
│   ├── pipeline.py                # Audio Pipeline Orchestrator
│   ├── deepfake_model.py          # Wav2Vec2 Model Wrapper & Inference Engine
│   ├── vad.py                     # Silero Voice Activity Detection Trimmer
│   ├── asr.py                     # Whisper-Tiny Speech-to-Text Transcriber
│   ├── audio_loader.py            # Audio Ingestion & Resampling (16kHz mono)
│   ├── EXPLAINER.md               # Technical Deep-Dive on Audio Architecture
│   ├── calibration/               # Audio Model Config & Threshold Tiers
│   └── tests/                     # Unit Tests for Audio Components
│
└── text_forensics/                # Text Forensics Subsystem
    ├── app.py                     # Standalone Text Forensics Streamlit App
    ├── pipeline.py                # Text Pipeline Orchestrator & Feature Extraction
    ├── feature_extractor.py       # Core Feature Calculators (Curvature, Entropy, etc.)
    ├── fusion.py                  # Logistic Regression Ensemble & Calibration
    ├── robust_curvature.py        # SmolLM2 Curvature Engine (Fast-DetectGPT)
    ├── TEXT_FORENSICS_MASTER_DOC.md # Technical Deep-Dive on Text Signals
    ├── calibration/               # Genre-Diverse Calibration Weights & Datasets
    └── tests/                     # Unit Tests for Text Components
```

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.9+**
- **FFmpeg** (Required for audio format decoding and Whisper ASR)
  - *Windows*: `winget install FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org/)
  - *Linux*: `sudo apt install ffmpeg`
  - *macOS*: `brew install ffmpeg`

### 1. Clone the Repository
```bash
git clone https://github.com/shruti0890/AI-audio-Text-Detection.git
cd AI-audio-Text-Detection
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLP Language Resources
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## 💻 How to Run

### Option 1: Unified Web Application (Recommended)
Launch the unified multi-modal web interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. 

#### Available Modes in Dashboard:
- 📝 **Text Only**: Analyze raw written text for AI generation signals.
- 🎙️ **Audio Only**: Upload an audio file or record live speech to evaluate voice authenticity and extract transcripts.
- ⚡ **Combined (Dual-Modality)**: Upload audio and text simultaneously to run complete cross-modal conflict detection.

---

### Option 2: Standalone Subsystem Dashboards
Run individual modality interfaces independently:

```bash
# Run Standalone Audio Forensics Dashboard
streamlit run audio_forensics/app.py

# Run Standalone Text Forensics Dashboard
streamlit run text_forensics/app.py
```

---

### Option 3: Python API Usage

#### Unified Fusion API
```python
from fusion_integration import run_full_pipeline

# Analyze both Audio and Text together
result = run_full_pipeline(
    text="Delve into this comprehensive summary outlining pivotal paradigm shifts.",
    audio_path="sample_audio.wav"
)

print(f"Audio Score   : {result['audio_score']}% ({result['audio_verdict']})")
print(f"Text Score    : {result['text_score']}% ({result['text_verdict']})")
print(f"Unified Verdict: {result['unified_verdict']}")
```

#### Cross-Modal Evaluation API
```python
from cross_modal_reasoner import evaluate_cross_modality

audio_res = {"audio_score": 78.5, "decision_threshold_pct": 56.0}
text_res  = {"ai_score": 18.2, "verdict": "Human"}

evaluation = evaluate_cross_modality(audio_res, text_res)

print(f"State       : {evaluation['classification']}")  # AI_VOICE_HUMAN_TEXT_CONFLICT
print(f"Title       : {evaluation['title']}")           # ⚠️ MODALITY CONFLICT
print(f"Description : {evaluation['description']}")
```

---

## 📊 Thresholds & Decision Tiers

### Audio Forensics Decision Tiers
| Score Range (%) | Classification Tier |
| :---: | :--- |
| **0.0 – 35.0%** | Authentic Human Voice |
| **35.1 – 55.0%** | Likely Human Voice |
| **55.1 – 74.0%** | Likely AI Voice |
| **74.1 – 100.0%** | Authentic AI Voice |

*(Decision Threshold: **56.0%**)*

### Text Forensics Decision Tiers
| Score Range (%) | Classification Tier |
| :---: | :--- |
| **0.0 – 20.0%** | Authentic Human Text |
| **20.1 – 44.9%** | Likely Human Text |
| **45.0 – 69.9%** | Likely AI Text |
| **70.0 – 100.0%** | Authentic AI Text |

*(Decision Threshold: **45.0%**)*

---

## 🧪 Testing & Verification

Run the automated Pytest test suites across both audio and text subsystems:

```bash
# Run all tests
pytest

# Run text forensics tests
pytest text_forensics/tests

# Run audio forensics tests
pytest audio_forensics/tests
```

---

## 📚 Technical Documentation & References

For deep technical details, mathematical formulations, and calibration benchmarks, refer to the in-repo documentation files:
- 📖 [**INTEGRATION_MASTER_DOC.md**](INTEGRATION_MASTER_DOC.md): Unified integration architecture, data schemas, and Q&A reference.
- 📖 [**TECHNICAL_ARCHITECTURE_AND_SYSTEM_EXPLAINER.md**](TECHNICAL_ARCHITECTURE_AND_SYSTEM_EXPLAINER.md): End-to-end technical system explainer and architectural reference.
- 📖 [**EVALUATION_METRICS_AND_BENCHMARKS.md**](EVALUATION_METRICS_AND_BENCHMARKS.md): Performance benchmarks, ROC-AUC metrics, and test dataset evaluations.
- 📖 [**AUDIO_CALIBRATION_PROTOCOL.md**](AUDIO_CALIBRATION_PROTOCOL.md): Audio feature calibration methodology and decision boundary tuning.
- 📖 [**audio_forensics/EXPLAINER.md**](audio_forensics/EXPLAINER.md): Complete explainer on Wav2Vec2 model, VAD, ASR, and audio pipeline.
- 📖 [**text_forensics/TEXT_FORENSICS_MASTER_DOC.md**](text_forensics/TEXT_FORENSICS_MASTER_DOC.md): Statistical formulas for curvature, burstiness, entropy, and logistic regression fusion.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<p align="center">
  <b>Developed for AI Audio & Text Forensics Research</b><br>
  <i>Detecting Synthetic Media • Empowering Digital Trust</i>
</p>
