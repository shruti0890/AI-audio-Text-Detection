# Audio & Cross-Modal Forensics System — Architecture & Explainer

---

## Table of Contents
1. [System Overview & Architecture](#1-system-overview--architecture)
2. [Audio Ingestion & Normalization (`audio_loader.py`)](#2-audio-ingestion--normalization-audio_loaderpy)
3. [Voice Activity Detection (`vad.py`)](#3-voice-activity-detection-vadpy)
4. [Speech-to-Text Transcription (`asr.py`)](#4-speech-to-text-transcription-asrpy)
5. [Deepfake Voice Classification (`deepfake_model.py`)](#5-deepfake-voice-classification-deepfake_modelpy)
6. [Threshold Calibration & Decision Tiers (`model_config.json`)](#6-threshold-calibration--decision-tiers-model_configjson)
7. [Audio Forensics Pipeline Orchestrator (`pipeline.py`)](#7-audio-forensics-pipeline-orchestrator-pipelinepy)
8. [Live Recording Architecture & Lifecycle](#8-live-recording-architecture--lifecycle)
9. [Transcribed Text Forensics (`text_forensics/`)](#9-transcribed-text-forensics-text_forensics)
10. [Stage 2 — Cross-Modality Reasoner (`cross_modal_reasoner.py`)](#10-stage-2--cross-modality-reasoner-cross_modal_reasonerpy)
11. [Web Application Interface (`app.py`)](#11-web-application-interface-apppy)
12. [Verification, Testing & Running the System](#12-verification-testing--running-the-system)

---

## 1. System Overview & Architecture

This system provides a unified forensic pipeline for detecting synthetic/AI-generated voices, AI-generated text, and cross-modal discrepancies (such as a genuine human voice reading an LLM-generated script, or an AI voice cloning human-written text).

### High-Level Data Flow

```
                                  USER INPUT
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
           📁 Upload Audio File                  🎤 Live Recording
         (.wav, .mp3, .flac, etc.)              (Browser MediaRecorder)
                    │                                     │
                    │                             Temporary Audio
                    │                           (live_recording_<uuid>)
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                       COMMON AUDIO FORENSICS PIPELINE
                                       │
     ┌─────────────────────────────────┼─────────────────────────────────┐
     ▼                                 ▼                                 ▼
1. Audio Loader & Normalization   2. Silero VAD (Silence Trimming)  3. Whisper-Tiny ASR
  (16kHz mono, PyAV / soundfile)   (Active human speech extraction)   (English Speech-to-Text)
     │                                 │                                 │
     └─────────────────────────────────┼─────────────────────────────────┘
                                       ▼
                       4. wav2vec2 Deepfake Classifier
                       (Sliding Window, T=1.15, Max Risk)
                                       │
                                       ▼
                             AUDIO FORENSIC OUTPUT
                        • Audio Score (0–100%)
                        • Decision Threshold (56.0%)
                        • 4-Tier Verdict (Human / AI)
                        • Generated Transcript
                                       │
                                       ▼ (User Clicks)
                        [ 🔬 Analyse Transcribed Text ]
                                       │
                                       ▼
                       5. FIVE-FEATURE TEXT FORENSICS
                       (Curvature + Burstiness + Entropy
                        + Regularity + Cliché Density)
                        • Text Score (0–100%)
                        • Decision Threshold (45.0%)
                        • Text Verdict (Human / AI)
                                       │
                                       ▼
                      6. CROSS-MODALITY REASONER
                      (Deterministic Matrix Evaluation)
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             ▼                                                   ▼
   ✓ MODALITIES CONSISTENT                             ⚠️ MODALITY CONFLICT
   • HUMAN_HUMAN (Real Voice + Real Text)             • HUMAN_VOICE_AI_TEXT (Human reading LLM script)
   • AI_AI (Synthetic Voice + AI Text)                • AI_VOICE_HUMAN_TEXT (TTS voice clone of human writing)
```

---

## 2. Audio Ingestion & Normalization (`audio_loader.py`)

### File: `audio_forensics/audio_loader.py`

**What this file is responsible for**:
Robustly ingesting audio from diverse container formats and codecs (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`), converting multi-channel audio to mono, resampling strictly to 16,000 Hz float32 arrays, and extracting audio metadata.

**Why this is critical**:
Downstream models (Silero VAD, Whisper-Tiny, and wav2vec2) were trained strictly on **16kHz single-channel mono speech**. Passing incorrect sampling rates or raw compressed streams leads to distorted frequency features and model crashes.

**Step-by-step logic**:
1. **Multi-Engine Decoding**:
   - Primary: Uses `av` (PyAV / FFmpeg bindings) or `soundfile` to read audio streams in memory without spawning external CLI processes.
   - Fallback: Uses `torchaudio.load` if PyAV fails.
2. **Channel Downmixing**:
   - Stereo / multi-channel arrays `(C, N)` are averaged across channels to produce a 1D mono array `(N,)`.
3. **Resampling**:
   - If the input sampling rate differs from 16,000 Hz, it applies polyphase filter resampling via `torchaudio.transforms.Resample` or `librosa.resample` to 16kHz.
4. **Metadata Extraction**:
   - Captures `original_format`, `original_duration_seconds`, `original_sample_rate`, `original_channels`, and `normalized_sample_rate`.

**Output**:
- `waveform`: 1D `numpy.ndarray` (float32, 16kHz mono).
- `metadata`: Dictionary containing audio properties.

---

## 3. Voice Activity Detection (`vad.py`)

### File: `audio_forensics/vad.py`

**What this file is responsible for**:
Identifies human speech timestamps and strips out prolonged silences, room tone, and non-speech background noise.

**Step-by-step logic**:
1. **Model Caching**: Loads Silero VAD (`snakers4/silero-vad`) from PyTorch Hub once, caching the neural network globally in `_VAD_MODEL`.
2. **Inference (`get_speech_timestamps`)**: Evaluates the 16kHz waveform using Silero's Recurrent Neural Network (RNN) with Short-Time Fourier Transform (STFT) features at a speech probability threshold of 0.5 (50%).
3. **Chunk Extraction & Assembly (`collect_chunks`)**: Concatenates all detected active speech frames into a continuous 1D float32 array `processed_audio`.
4. **Safety Fallback**: If an audio file contains pure silence or no detectable speech, it safely falls back to returning the normalized input waveform rather than returning an empty array.

**Output**:
- `processed_audio`: Cleaned 16kHz float32 mono speech waveform.

---

## 4. Speech-to-Text Transcription (`asr.py`)

### File: `audio_forensics/asr.py`

**What this file is responsible for**:
Transcribes spoken speech into English text using OpenAI's Whisper-Tiny (`openai/whisper-tiny`) model via Hugging Face `transformers`.

**Why this transcript is essential**:
1. **Human Auditability**: Allows analysts and users to inspect the exact spoken content.
2. **Cross-Modality Integration**: Supplies the text that is passed to the Five-Feature Text Forensics pipeline to determine whether the spoken passage was generated by an LLM.

**Step-by-step logic**:
1. **Model Caching**: Initializes `pipeline("automatic-speech-recognition", model="openai/whisper-tiny")` once into `_ASR_PIPELINE`.
2. **In-Memory Feed**: Feeds the 16kHz numpy array directly as `{"raw": waveform, "sampling_rate": 16000}`, bypassing external disk writes.
3. **Token Decoding**: Whisper computes log-mel spectrograms, predicts autoregressive language tokens, and decodes them into plain English sentences.
4. **Contract Simplification**: Extracts the clean string (`result["text"]`), discarding auxiliary metadata.

**Output**:
- `transcript`: Plain Python string (`str`).

---

## 5. Deepfake Voice Classification (`deepfake_model.py`)

### File: `audio_forensics/deepfake_model.py`

**What this file is responsible for**:
Calculates acoustic deepfake risk by analyzing raw speech waveforms with `garystafford/wav2vec2-deepfake-voice-detector`.

**Step-by-step logic**:

### A. Offline-First Model Initialization & Label Resolution
1. **Offline Loading**: Initializes `AutoModelForAudioClassification` with `local_files_only=True` first, avoiding slow network timeouts (~60s saved) on repeated startups.
2. **Dynamic Label Index Resolution**: Inspects `model.config.id2label` (e.g. `{0: 'real', 1: 'fake'}`) to resolve `_FAKE_IDX` and `_REAL_IDX` dynamically. It **never** hardcodes indices, preventing label inversion bugs.
3. **Audit Verification**: Persists verified index mappings to `audio_forensics/calibration/model_config.json`.

### B. Sliding Window Segmentation & Batching
1. **Length Guard**: Clips long audio to a maximum of 30.0s (`480,000` samples) for memory safety.
2. **Window Parameters**:
   - Window Size: `5.0s` (`80,000` samples).
   - Overlap: `1.0s` (`16,000` samples).
   - Step Size: `4.0s` (`64,000` samples).
3. **Short Audio Handling**: Audio $<5.0\text{s}$ is evaluated as a single padded window.
4. **Batched Inference**: All windows are extracted and processed in a single forward pass through the transformer feature extractor and encoder.

### C. Temperature-Scaled Sigmoid Scoring Formula
The model outputs raw pre-softmax logits for each window $i$: $\text{logit}_{fake}^{(i)}$ and $\text{logit}_{real}^{(i)}$.

Instead of standard softmax (which compresses differences), the system computes the logit difference scaled by a calibrated temperature factor $T = 1.15$:

$$\Delta^{(i)} = \text{logit}_{fake}^{(i)} - \text{logit}_{real}^{(i)}$$

$$S_{Audio}^{(i)} = \text{Sigmoid}\left(\frac{\Delta^{(i)}}{T}\right) \times 100 = \frac{1}{1 + e^{-\frac{\Delta^{(i)}}{T}}} \times 100$$

### D. Aggregation Strategy: `max_risk`
The overall audio deepfake score is assigned using the **`max_risk`** strategy (the score of the highest-risk window):

$$S_{Audio} = \max_{i} S_{Audio}^{(i)}$$

The mean and median window scores are also computed and exposed in the audit metadata.

**Output Schema**:
```python
{
    "s_audio": float,               # 0.0 to 100.0 (overall deepfake risk score)
    "logit_fake": float,             # Logit from highest-risk window
    "logit_real": float,             # Logit from highest-risk window
    "decision_threshold": 56.0,      # Calibrated decision boundary
    "temperature": 1.15,             # Logit temperature scaling factor
    "n_windows": int,                # Total windows analyzed
    "window_scores": list[float],    # Scores for all windows
    "mean_score": float,             # Mean window score
    "median_score": float,           # Median window score
    "confidence_tiers": dict,        # Margin breakpoints
}
```

---

## 6. Threshold Calibration & Decision Tiers (`model_config.json`)

### File: `audio_forensics/calibration/model_config.json` & `run_calibration.py`

Empirical calibration was conducted using Receiver Operating Characteristic (ROC) curve analysis on ground-truth real human voices and synthetic TTS deepfake audio clips.

### A. Decision Threshold
- **Binary Decision Boundary**: **`56.0%`**
  - Scores $\ge 56.0\%$ are classified as **`AI`** (Synthetic Voice).
  - Scores $< 56.0\%$ are classified as **`HUMAN`** (Natural Human Voice).

### B. 4-Tier Verdict Classification System
| Score Range | Verdict Tier | Description |
| :--- | :--- | :--- |
| **0.0% – 35.0%** | 🟢 **Authentic Human Voice** | High confidence genuine human vocal tract resonance and natural breath patterns. |
| **36.0% – 55.0%** | 🟢 **Likely Human Voice** | Moderate confidence human speech acoustics (<56.0% decision threshold). |
| **56.0% – 74.0%** | 🟡 **Likely AI Voice** | Moderate confidence synthetic acoustic cues or vocoder artifacts (≥56.0% threshold). |
| **75.0% – 100.0%** | 🔴 **Authentic AI Voice** | High confidence synthetic voice synthesis / TTS deepfake. |

### C. Logit Margin Confidence Tiers
Based on the absolute logit difference $|\Delta| = |\text{logit}_{fake} - \text{logit}_{real}|$:
- $|\Delta| > 3.0$: **Extreme Confidence**
- $|\Delta| > 1.5$: **High Confidence**
- $|\Delta| > 0.5$: **Moderate Confidence**
- $|\Delta| \le 0.5$: **Borderline / Low Confidence**

---

## 7. Audio Forensics Pipeline Orchestrator (`pipeline.py`)

### File: `audio_forensics/pipeline.py`

Chains all audio analysis stages in memory and returns a contract-locked schema dictionary.

```python
analyze_audio(audio_path: str, batch_size: int = 1, progress_callback = None) -> dict
```

**Execution Sequence**:
1. `load_and_normalize_audio(audio_path)` $\to$ 16kHz mono float32 waveform.
2. `strip_silence(raw_waveform)` $\to$ VAD silence-trimmed speech.
3. `transcribe(processed_audio)` $\to$ Whisper-Tiny English transcript.
4. `score_audio(processed_audio)` $\to$ Batched wav2vec2 sliding-window classification.
5. Returns standard dictionary with timing benchmarks and confidence tiers.

---

## 8. Live Recording Architecture & Lifecycle

### Component: `app.py` (Audio Analysis $\to$ Live Recording)

Live microphone recording is fully integrated into the existing Audio Forensics architecture without creating duplicate scoring logic.

```
Browser Microphone → WebRTC MediaRecorder (st.audio_input)
                           ↓
               temp/live_recording_<uuid>.wav
                           ↓
               audio_forensics.pipeline.analyze_audio()
                           ↓
                   Display Results
                           ↓
         [ Record Again ] → Delete Temp Audio & Reset State
```

### Key Architectural Principles:
1. **Zero Pipeline Duplication**: Live recorded audio enters the exact same `audio_loader.py` $\to$ `vad.py` $\to$ `asr.py` $\to$ `deepfake_model.py` path as uploaded files.
2. **Browser-Native WebRTC Recording**: Uses `st.audio_input` (`MediaRecorder` API) with permission handling, live recording timers, waveform feedback, and instant playback preview.
3. **Isolated Temporary File Storage**: Each recording receives a unique temporary filepath (`temp/live_recording_<uuid>.wav`). Recordings are never permanently stored, added to calibration datasets, or committed to git.
4. **Lifecycle & Safe Cleanup**: Files remain accessible during Stage 1 and Stage 2 analysis, and are automatically deleted upon clicking **`[ 🔄 Record Again ]`**, switching inputs, or concluding the session.

---

## 9. Transcribed Text Forensics (`text_forensics/`)

### File: `text_forensics/pipeline.py`

When the user clicks **`[ 🔬 Analyse Transcribed Text ]`**, the generated transcript is evaluated using the **Five-Feature Logistic Regression Model**:

1. **Curvature (Fast-DetectGPT)**: Evaluates local log-probability curvature via `distilgpt2`. LLM text occupies negative curvature regions in model probability space.
2. **Burstiness**: Computes sentence-length variation ($\sigma/\mu$). Human writing exhibits high rhythmic variation, while LLMs produce uniform sentence lengths.
3. **Lexical Entropy**: Measures Shannon entropy and Type-Token Ratio (TTR).
4. **Structural Regularity**: Evaluates sentence starter diversity and POS tag n-gram overlap.
5. **Cliché Density**: Scans for 50+ overused AI transition words and phrases.

### Text Thresholds & Verdicts:
- **Decision Threshold**: **`45.0%`**
  - $\le 20\%$: Human
  - $20\% - 45\%$: Likely Human
  - $45\% - 70\%$: Likely AI (Classified as **`AI`**)
  - $\ge 70\%$: AI (Classified as **`AI`**)

---

## 10. Stage 2 — Cross-Modality Reasoner (`cross_modal_reasoner.py`)

### File: `cross_modal_reasoner.py`

**What this file is responsible for**:
Deterministically compares the **Audio Modality Classification** (from wav2vec2) with the **Transcribed Text Modality Classification** (from Five-Feature Text Forensics).

### The Four Deterministic Outcomes:

```
                                ┌─────────────────────────┐
                                │   Audio Classification  │
                                │   (Threshold: 56.0%)    │
                                └────────────┬────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
                 AUDIO = HUMAN                                 AUDIO = AI
                      │                                             │
         ┌────────────┴────────────┐                   ┌────────────┴────────────┐
         ▼                         ▼                   ▼                         ▼
    TEXT = HUMAN              TEXT = AI           TEXT = HUMAN              TEXT = AI
         │                         │                   │                         │
         ▼                         ▼                   ▼                         ▼
   HUMAN_HUMAN           HUMAN_VOICE_AI_TEXT     AI_VOICE_HUMAN_TEXT           AI_AI
 (✓ Consistent)              (⚠️ Conflict)           (⚠️ Conflict)         (✓ Consistent)
```

### Outcome Descriptions:

1. **`HUMAN_HUMAN` (✓ MODALITIES CONSISTENT)**
   - **Audio**: Human Voice ($<56.0\%$)
   - **Text**: Human Text ($<45.0\%$)
   - **Interpretation**: Both acoustic vocal tract characteristics and linguistic writing style indicate authentic human creation.

2. **`AI_AI` (✓ MODALITIES CONSISTENT)**
   - **Audio**: AI Voice ($\ge 56.0\%$)
   - **Text**: AI Text ($\ge 45.0\%$)
   - **Interpretation**: Both speech acoustics (synthetic voice synthesis) and linguistic style (LLM patterns) indicate artificial generation.

3. **`HUMAN_VOICE_AI_TEXT_CONFLICT` (⚠️ MODALITY CONFLICT)**
   - **Audio**: Human Voice ($<56.0\%$)
   - **Text**: AI Text ($\ge 45.0\%$)
   - **Interpretation**: Natural human voice detected, but the spoken text was likely authored by an AI language model (e.g., a human speaker reading an AI-generated script or teleprompter).

4. **`AI_VOICE_HUMAN_TEXT_CONFLICT` (⚠️ MODALITY CONFLICT)**
   - **Audio**: AI Voice ($\ge 56.0\%$)
   - **Text**: Human Text ($<45.0\%$)
   - **Interpretation**: Synthetic voice / voice cloning detected, but the underlying text exhibits natural human phrasing (e.g., AI voice cloning of human-written text).

---

## 11. Web Application Interface (`app.py`)

The main Streamlit interface in `app.py` presents two primary analysis modes:

### Sidebar Navigation
- **📝 Text Analysis**: Direct text input for standalone 5-Feature Text Forensics.
- **🎙️ Audio Analysis**: Unified audio workspace supporting two input methods:
  - `📁 Upload Audio File`: Multi-format audio upload (.wav, .mp3, .flac, .ogg, .m4a, .aac) with preview player.
  - `🎤 Live Recording`: Browser-native microphone recording with duration tracking, playback, and reset.

### UI Panels & Workflow
1. **Section 1 — Input**: Select input method and provide audio file or live recording.
2. **Section 2 — Execution**: Shows animated progress spinners for VAD, Whisper transcription, and wav2vec2 sliding-window scoring.
3. **Section 3 — Audio Results**:
   - 4-Tier Verdict Card with color-coded severity.
   - Deepfake Risk score, decision threshold (56.0%), logit metrics, and sliding-window stats.
   - Component timing breakdown (VAD, ASR, Deepfake scoring).
   - Whisper-Tiny Speech-to-Text transcript.
4. **Stage 2 — Cross-Modality Analysis**:
   - Interactive **`[ 🔬 Analyse Transcribed Text ]`** button (enabled when transcript $>30$ words).
   - Transcribed Text forensic evaluation (Text score, 45.0% threshold, verdict).
   - Cross-Modality comparison card with Consistent / Conflict badges.
   - Side-by-side audio vs text evidence comparison.
5. **Audit Inspector**: Expandable JSON viewers displaying raw model outputs, logits, and feature vectors.

---

## 12. Verification, Testing & Running the System

### Running the Application
From the repository root:
```bash
.\venv\Scripts\python.exe -m streamlit run app.py
```
Open in browser: `http://localhost:8501`

### Running the Test Suite
```bash
# Test Voice Activity Detection (VAD)
.\venv\Scripts\pytest.exe audio_forensics/tests/test_vad.py -v

# Test Automatic Speech Recognition (ASR Whisper)
.\venv\Scripts\pytest.exe audio_forensics/tests/test_asr.py -v

# Test wav2vec2 Deepfake Classifier
.\venv\Scripts\pytest.exe audio_forensics/tests/test_deepfake_model.py -v

# Test Full Audio Forensics Pipeline
.\venv\Scripts\pytest.exe audio_forensics/tests/test_audio_pipeline.py -v
```

### Running Threshold Recalibration
```bash
.\venv\Scripts\python.exe audio_forensics/calibration/run_calibration.py
```
