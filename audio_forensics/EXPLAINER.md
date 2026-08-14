# Audio Forensics Module Explainer & Progress Log

---

## Phase 2.1 — Environment & Skeleton

### File: `audio_forensics/vad.py`

> **Note**: `vad.py` was originally created as a stub in Phase 2.1 and was updated with full Silero VAD logic in [Phase 2.2](#phase-22--voice-activity-detection-silence-stripping).

**What this file is responsible for**:
This file handles Voice Activity Detection (VAD), which identifies human speech in audio recordings and removes non-speech or silent portions. Its single job in the pipeline is to clean raw audio inputs so downstream speech recognition and deepfake classification models operate exclusively on active human speech.

**Walkthrough of what happens, in order**:
* **What comes in**: An `audio_path` string pointing to an input audio file (such as a `.wav` or `.mp3` file) or live microphone recording.
* **Step-by-step logic**:
  1. *Audio Loading*: In the full implementation, it loads the audio file into memory.
  2. *Speech Detection*: It passes the audio through a VAD neural network (Silero VAD) to calculate timestamps where speech begins and ends.
  3. *Silence Removal & Resampling*: It trims away non-speech segments, concatenates speech segments, and resamples the audio to 16,000 Hz (16kHz sample rate) in single-channel mono. Resampling to 16kHz is mandatory because both downstream models (Whisper ASR and Wav2Vec2) were specifically pre-trained on 16kHz speech.
* **What comes out**: `processed_audio`, an array of float numbers representing the cleaned 16kHz speech waveform. This output is consumed next by `asr.py` for transcription and `deepfake_model.py` for deepfake classification.

**Key variables/objects worth knowing**:
* `processed_audio`: A 1D `numpy` array of float32 values sampled at 16kHz. Passing raw numerical arrays in memory between modules avoids slow disk read/write cycles from creating temporary audio files.

**Anything simplified/stubbed for now**:
In Phase 2.1, Silero VAD is not loaded yet. `strip_silence()` currently returns a stub 1D array of zeros representing 1 second of silent 16kHz audio (`np.zeros(16000, dtype=np.float32)`).

**How to verify this file works**:
Run `.\venv\Scripts\pytest.exe audio_forensics/tests/test_vad.py`. It calls `strip_silence("dummy_path.wav")` and asserts that the return value is a valid 1D `numpy.ndarray`.

---

### File: `audio_forensics/asr.py`

> **Note**: `asr.py` was originally created as a stub in Phase 2.1 and was updated with full Whisper-Tiny ASR logic in [Phase 2.3](#phase-23--speech-to-text-whisper-tiny).

**What this file is responsible for**:
This file handles Automatic Speech Recognition (ASR), which converts spoken voice audio into written text. Its single job is to transcribe speech so that the text transcript can be passed to text forensics modules and cross-modal consistency checks (which verify whether spoken audio matches surrounding text).

**Walkthrough of what happens, in order**:
* **What comes in**: `processed_audio`, which is the silence-stripped 16kHz speech array produced by `vad.py` (or a direct file path).
* **Step-by-step logic**:
  1. *Model Pipeline*: In the full implementation, it passes the audio into the Hugging Face `transformers` speech-to-text pipeline powered by OpenAI's Whisper-Tiny model.
  2. *Decoding*: Whisper processes acoustic frequency features, predicts word tokens, and decodes them into written English text sentences.
  3. *Contract Simplification*: It extracts the raw string text from the model's dictionary response, discarding extraneous metadata like word timestamps or log-probabilities.
* **What comes out**: A plain Python string containing the spoken text. This is returned to `pipeline.py` to be handed off to the top-level cross-modal fusion engine and text forensics module.

**Key variables/objects worth knowing**:
* `processed_audio`: The 16kHz audio waveform array input.
* Return `str`: A simple unadorned string containing the transcript. Returning a clean string keeps the interface contract straightforward for external consumers.

**Anything simplified/stubbed for now**:
In Phase 2.1, Whisper-Tiny is not loaded yet. `transcribe()` currently returns a placeholder string: `"This is a placeholder transcript from the Whisper-Tiny ASR stub."`.

**How to verify this file works**:
Run `.\venv\Scripts\pytest.exe audio_forensics/tests/test_asr.py`. It passes a dummy audio array to `transcribe()` and verifies that it returns a non-empty string.

---

### File: `audio_forensics/deepfake_model.py`

> **Note**: `deepfake_model.py` was originally created as a stub in Phase 2.1 and was updated with full wav2vec2 inference logic in [Phase 2.4](#phase-24--deepfake-classification-wav2vec2-deepfake-voice-detector).

**What this file is responsible for**:
This file performs neural network deepfake classification on audio recordings. Its single job is to analyze voice acoustic patterns and calculate numerical indicators determining whether a voice is human (real) or AI-synthesized (fake).

**Walkthrough of what happens, in order**:
* **What comes in**: `processed_audio`, the silence-stripped 16kHz speech waveform array output by `vad.py`.
* **Step-by-step logic**:
  1. *Feature Extraction & Inference*: In the full implementation, it passes the audio through `garystafford/wav2vec2-deepfake-voice-detector`.
  2. *Logit Extraction*: The model emits unnormalized output numbers called **logits** for two classes: real voice (`logit_real`) and synthetic voice (`logit_fake`). Logits are the raw scores calculated by the neural network's final layer prior to probability conversion.
  3. *Score Calculation*: Instead of standard probability softmax, we calculate the difference between logits (`logit_fake - logit_real`) and pass it into a **Sigmoid** mathematical function: `S_Audio = Sigmoid(logit_fake - logit_real) * 100`. The Sigmoid function smoothly squashes any positive or negative difference value into a clean 0 to 100 percentage score (0 = definitely human, 100 = definitely deepfake).
* **What comes out**: A dictionary containing `{"s_audio": float, "logit_fake": float, "logit_real": float}` consumed next by `pipeline.py`.

**Key variables/objects worth knowing**:
* `logit_fake` & `logit_real`: Unnormalized raw neural network output scores.
* `s_audio`: The final audio deepfake risk score scaled from 0.0 to 100.0.

**Anything simplified/stubbed for now**:
In Phase 2.1, the `wav2vec2` model is not loaded yet. `score_audio()` returns neutral stub values: `logit_fake = 0.0`, `logit_real = 0.0`, and `s_audio = 50.0`.

**How to verify this file works**:
Run `.\venv\Scripts\pytest.exe audio_forensics/tests/test_deepfake_model.py`. It verifies that `score_audio()` returns a dictionary containing keys `'s_audio'`, `'logit_fake'`, and `'logit_real'`.

---

### File: `audio_forensics/pipeline.py`

**What this file is responsible for**:
This file serves as the main entrypoint and orchestrator for the entire audio forensics engine. Its single job is to execute VAD, ASR, and deepfake classification in sequence on an input file, assembling the results into the standard JSON dictionary schema required by `fusion.py`.

**Walkthrough of what happens, in order**:
* **What comes in**: `audio_path`, a string filepath pointing to an input audio file.
* **Step-by-step logic**:
  1. *Silence Removal*: First, it calls `strip_silence(audio_path)` from `vad.py` to extract a clean 16kHz speech waveform `processed_audio`.
  2. *Transcription*: Second, it passes `processed_audio` to `transcribe(processed_audio)` in `asr.py` to generate the text transcript.
  3. *Deepfake Scoring*: Third, it passes `processed_audio` to `score_audio(processed_audio)` in `deepfake_model.py` to compute deepfake logits and the `s_audio` score.
  4. *Schema Assembly*: Fourth, it merges all results into a standardized dictionary with key names matching the exact schema contract expected by the top-level fusion engine.
* **What comes out**: A dictionary formatted as:
  ```json
  {
    "audio_score": float,
    "logit_fake": float,
    "logit_real": float,
    "transcript": str,
    "wer_confidence_note": str
  }
  ```
  This output is consumed by `fusion.py` for cross-modal score merging and conformal risk tagging.

**Key variables/objects worth knowing**:
* `analyze_audio(audio_path)`: The single public function exposed to external callers (`fusion.py`). Keeping its signature and return schema stable allows integration work to proceed independently of internal model development.

**Anything simplified/stubbed for now**:
In Phase 2.1, because `vad.py`, `asr.py`, and `deepfake_model.py` return stub values, `analyze_audio` returns placeholder values. However, the data flow, function calls, return keys, and types are 100% complete and match the final schema contract.

**How to verify this file works**:
Run `.\venv\Scripts\pytest.exe audio_forensics/tests/test_audio_pipeline.py` or run `.\venv\Scripts\python.exe -c "from audio_forensics.pipeline import analyze_audio; print(analyze_audio('clip.wav'))"`. It executes the full pipeline and outputs the structured dictionary.

---

## Phase 2.2 — Voice Activity Detection (Silence Stripping)

### File: `audio_forensics/vad.py`

**What this file is responsible for**:
This file replaces the Phase 2.1 stub with a fully functioning Voice Activity Detection (VAD) engine powered by Silero VAD (`snakers4/silero-vad`). Its single job is to ingest raw audio files or waveforms, detect active human speech timestamps, strip away silence and background noise, and return a clean 16kHz float32 mono audio array for downstream models.

**Walkthrough of what happens, in order**:
* **What comes in**: An `audio_input` parameter, which can be an audio file path string (`.wav`, `.mp3`), a 1D/2D `numpy.ndarray`, or a PyTorch `Tensor`.
* **Step-by-step logic**:
  1. *Model Caching (`get_vad_model`)*: Loads Silero VAD from PyTorch Hub (`snakers4/silero-vad`) with `skip_validation=True` and caches the model globally in `_VAD_MODEL` so it is loaded into memory only once rather than re-downloaded/re-initialized on every audio clip.
  2. *Audio Ingestion & Resampling*: If given a file path, `librosa.load` reads the file, converts stereo channels to mono by averaging, and resamples the audio to 16,000 Hz (16kHz). If given numpy/tensor arrays, stereo channels are merged into 1D mono float32 format.
  3. *VAD Inference (`get_speech_timestamps`)*: The waveform tensor is passed into Silero VAD with a speech probability confidence threshold of 0.5 (50% confidence). Silero VAD uses a light Recurrent Neural Network (RNN) with STFT spectrogram features to return a list of dictionary timestamps `[{'start': sample_idx, 'end': sample_idx}, ...]`.
  4. *Segment Trimming & Concatenation (`collect_chunks`)*: If speech timestamps are found, `collect_chunks` extracts the audio slices corresponding to speech and concatenates them into a single continuous tensor. If no speech is detected (e.g. a pure silence file), it safely falls back to returning the original waveform rather than crashing on empty arrays.
* **What comes out**: A 1D float32 `numpy.ndarray` containing only active speech samples at 16kHz mono, ready to be passed directly to `asr.py` and `deepfake_model.py`.

**Key variables/objects worth knowing**:
* `sampling_rate=16000`: Standard speech sample rate required by Whisper and Wav2Vec2.
* `threshold=0.5`: The confidence score above which Silero VAD marks a frame as active speech.
* `speech_timestamps`: List of dicts specifying sample start/end positions for detected speech.

**Anything simplified/stubbed for now**:
Nothing is stubbed in `vad.py` anymore. Silence detection, audio resampling, format normalization, and chunk concatenation are 100% functional.

**How to verify this file works**:
Run `.\venv\Scripts\pytest.exe audio_forensics/tests/test_vad.py`. It runs silence stripping against synthetic audio clips, file paths, and stereo arrays to verify array dimensions, float32 types, and timestamp extraction.

---

### File: `audio_forensics/tests/test_vad.py`

**What this file is responsible for**:
This test module validates `audio_forensics/vad.py` against realistic audio inputs, ensuring silence stripping works correctly for file paths, numpy arrays, and multi-channel stereo waveforms.

**Walkthrough of what happens, in order**:
* **What comes in**: Test triggers from PyTest.
* **Step-by-step logic**:
  1. *Setup (`setup_module`)*: Generates a test WAV file (`test_clip_with_silence.wav`) containing 1.5s silence, 2s active tone signal, and 1.5s silence in `sample_clips/`.
  2. *File Test (`test_strip_silence_from_filepath`)*: Calls `strip_silence(path)` on the test clip and asserts that it returns a valid 1D float32 array.
  3. *Array Test (`test_strip_silence_from_numpy_array`)*: Passes a raw 1D float32 numpy waveform array directly to `strip_silence`.
  4. *Stereo Test (`test_strip_silence_stereo_conversion`)*: Passes a 2D stereo numpy array `(2, 16000)` and asserts that it is correctly flattened to 1D mono.
* **What comes out**: Test pass/fail assertions.

**Key variables/objects worth knowing**:
* `SAMPLE_CLIP_PATH`: Filepath to the test audio clip in `sample_clips/`.

**Anything simplified/stubbed for now**:
Fully functional unit tests covering file paths, raw numpy arrays, and stereo conversion.

**How to verify this file works**:
Run `.\venv\Scripts\pytest.exe audio_forensics/tests/test_vad.py` directly.

---

## Phase 2.3 — Speech-to-Text (Whisper-Tiny)

### File: `audio_forensics/asr.py`

**What this file is responsible for**:
This file converts spoken audio waveforms into written text using OpenAI's Whisper-Tiny model (`openai/whisper-tiny`) via the Hugging Face `transformers` ASR pipeline. Its single job in the pipeline is to generate a clean, unadorned text transcript from silence-stripped audio.

**Walkthrough of what happens, in order**:
* **What comes in**: `processed_audio`, which is the 16kHz float32 mono numpy array output by `vad.py` (or a direct filepath string / PyTorch tensor).
* **Step-by-step logic**:
  1. *Model Caching (`get_asr_pipeline`)*: Loads `pipeline("automatic-speech-recognition", model="openai/whisper-tiny")` and caches it in `_ASR_PIPELINE` to avoid reloading model weights on repeated invocations.
  2. *Input Normalization*: Converts input file paths or tensors into in-memory `{"raw": waveform, "sampling_rate": 16000}` numpy dictionary structures using `librosa.load`. Pre-loading audio directly into numpy arrays prevents Hugging Face transformers from attempting to invoke external `ffmpeg` binary executables on Windows systems.
  3. *Whisper Inference*: Passes the raw float32 array into Whisper-Tiny, which computes log-mel spectrogram features and decodes spoken tokens into English text.
  4. *Contract Simplification & Metadata Stripping*: Extracts the raw string value from the pipeline's output dictionary (`result["text"]`), discarding extraneous timestamps and log-probabilities.
* **What comes out**: A plain Python string (`str`) containing the spoken text.

**Why this transcript matters for downstream consumers**:
The returned text transcript serves two critical downstream architectural consumers:
1. **Text Forensics Pipeline (`text_forensics/`)**: The teammate's text pipeline analyzes the generated transcript for lexical entropy, burstiness, curvature, and cliches to determine if the spoken text itself exhibits LLM-generated patterns.
2. **Phase 3 Cross-Modal Consistency Check**: The cross-modal checker compares the transcript extracted from audio against any user-supplied text to verify whether the spoken voice matches the expected textual claim (detecting audio replacement / voice spoofing mismatches).

**Key variables/objects worth knowing**:
* `_ASR_PIPELINE`: Cached Hugging Face ASR pipeline object storing Whisper-Tiny weights in RAM.
* Return `str`: Unwrapped plain string. Returning a plain string enforces a simple, stable contract without leaking pipeline-internal dictionary wrappers to `fusion.py`.

**Anything simplified/stubbed for now**:
Nothing is stubbed in `asr.py`. Model initialization, pre-loading via librosa, Whisper-Tiny decoding, and string extraction are 100% functional.

**How to verify this file works**:
Run `.\venv\Scripts\pytest.exe audio_forensics/tests/test_asr.py`.

---

### File: `audio_forensics/tests/test_asr.py`

**What this file is responsible for**:
This test module validates `audio_forensics/asr.py` against both synthetic waveforms and real/synthetic spoken speech audio clips, verifying that transcription produces plain strings rather than wrapped dictionaries or metadata objects.

**Walkthrough of what happens, in order**:
* **What comes in**: PyTest test execution triggers.
* **Step-by-step logic**:
  1. *Setup (`setup_module`)*: Ensures sample audio files exist in `sample_clips/`.
  2. *Numpy Array Test (`test_transcribe_numpy_array`)*: Passes a 16kHz float32 numpy waveform into `transcribe()` and verifies the return type is a plain `str` and not a `dict`.
  3. *Filepath Test (`test_transcribe_filepath`)*: Passes a `.wav` file path into `transcribe()` and verifies type compliance.
  4. *Spoken Speech Test (`test_transcribe_speech_clip`)*: Passes `speech_sample.wav` (synthetic spoken voice) into `transcribe()` and asserts that the returned transcript is a non-empty string matching the spoken content.
* **What comes out**: Test pass/fail assertions.

**Key variables/objects worth knowing**:
* `SPEECH_CLIP_PATH`: Path to `sample_clips/speech_sample.wav`.

**Anything simplified/stubbed for now**:
Fully functional unit tests covering numpy arrays, audio file paths, and spoken speech clips.

**How to verify this file works**:
Run `.\venv\Scripts\pytest.exe audio_forensics/tests/test_asr.py` directly.

---

## Phase 2.4 — Deepfake Classification (wav2vec2-deepfake-voice-detector)

### File: `audio_forensics/deepfake_model.py`

**What this file is responsible for**:
This file loads `garystafford/wav2vec2-deepfake-voice-detector` from Hugging Face and uses it to classify a voice waveform as real (human) or fake (AI-synthesized). Its single job is to produce a 0–100 deepfake confidence score (`s_audio`) plus the raw per-class logit values that back it up — these are the numbers `pipeline.py` and `fusion.py` ultimately consume.

**Walkthrough of what happens, in order**:
* **What comes in**: `processed_audio`, which is the 16kHz float32 mono numpy array from `vad.py`, a file path string, or a PyTorch tensor.
* **Step-by-step logic**:
  1. *Lazy Model Loading (`_load_model`)*: On the very first call to `score_audio()`, it downloads and caches `garystafford/wav2vec2-deepfake-voice-detector` using Hugging Face `AutoModelForAudioClassification` and `AutoFeatureExtractor`. Both model weights and the processor are stored in module-level globals (`_MODEL`, `_PROCESSOR`) so they are only loaded once per Python process — not once per audio clip.
  2. *Label Index Resolution (CRITICAL)*: Immediately after loading, the code inspects `model.config.id2label` (e.g. `{0: 'real', 1: 'fake'}`). It resolves which integer index corresponds to `'fake'` and which to `'real'` by searching that dictionary — it **never** assumes `0 = real`. This prevents silent score inversion if a future model checkpoint swaps the label order.
  3. *Calibration Persistence*: The verified label mapping is written to `calibration/model_config.json` with `"status": "verified"`. This file is version-controlled, making the mapping auditable in git history.
  4. *Waveform Normalization*: File paths are loaded via `librosa.load` (same strategy as `asr.py`) to avoid an external `ffmpeg` dependency on Windows. Tensors are detached and converted to numpy. Stereo arrays are averaged to mono.
  5. *30-Second Truncation*: The waveform is clipped to `30s × 16000 samples/s = 480,000 samples` before feature extraction. wav2vec2 is transformer-based and its memory scales with sequence length — very long recordings can exhaust GPU/CPU RAM without this guard.
  6. *Feature Extraction*: `AutoFeatureExtractor` pads and normalises the waveform into a `(1, sequence_length)` tensor that wav2vec2 expects as input.
  7. *Inference — Raw Logits, Not Probabilities*: The model runs in `torch.no_grad()` mode (disabling gradient tracking for efficiency). The output is `outputs.logits`, shape `(1, 2)` — **these are pre-softmax values** (i.e., logits). We deliberately do NOT use `softmax(logits)` probabilities here, because softmax normalises outputs to sum to 1, which compresses the relative difference and distorts the sigmoid-of-difference formula.
  8. *Scoring Formula*: `diff = logit_fake - logit_real`. `S_Audio = Sigmoid(diff) × 100`. The Sigmoid function maps any real-valued difference into `(0, 1)`, and multiplying by 100 gives a human-readable percentage. A score near 100 means the model is highly confident the voice is synthetic; near 0 means it is confident it is real human speech.
* **What comes out**: A dict `{"s_audio": float, "logit_fake": float, "logit_real": float}` consumed by `pipeline.py`.

**Key variables/objects worth knowing**:
* `_FAKE_IDX` / `_REAL_IDX`: Module globals set once at load time from `id2label`. Everything downstream uses these, not hardcoded integers.
* `logits[0, _FAKE_IDX]` / `logits[0, _REAL_IDX]`: The two raw scalar values extracted from the model's final layer.
* `_MAX_SECONDS = 30`: Hard ceiling on input length. This is not a quality threshold — it is purely a memory safety guard.
* `calibration/model_config.json`: Written at model load time. Contains `id2label`, `fake_index`, `real_index`, and `status`. If `status != "verified"`, the model has not been properly loaded yet.

**Anything simplified/stubbed for now**:
Nothing is stubbed. Label resolution, calibration persistence, waveform normalization, feature extraction, logit extraction, and sigmoid scoring are all fully functional.

**How to verify this file works**:
Run `.\.venv\Scripts\pytest.exe audio_forensics/tests/test_deepfake_model.py`. Check that `calibration/model_config.json` shows `"status": "verified"` after the run.

---

### File: `audio_forensics/calibration/run_calibration.py`

**What this file is responsible for**:
This script executes empirical threshold calibration for the audio deepfake detector (`garystafford/wav2vec2-deepfake-voice-detector`). Its single job is to process directories of real and synthetic voice clips, calculate per-clip scores, generate ROC (Receiver Operating Characteristic) curves, calculate Equal Error Rate (EER) and conservative operating points, and calibrate the 4-tier decision threshold system.

**Walkthrough of what happens, in order**:
* **What comes in**: Audio clips located in `audio_forensics/sample_clips/fake/` (TTS/AI generated) and `audio_forensics/sample_clips/real/` (authentic human voice).
* **Step-by-step logic**:
  1. *Dataset Scoring (`_score_directory`)*: Iterates through files in `fake/` (label 1) and `real/` (label 0) directories, passing each file to `pipeline.analyze_audio()` to extract `audio_score` and raw logits.
  2. *ROC Curve Generation (`_compute_roc`)*: Sweeps decision thresholds from 0.0% to 100.0% in 0.5% increments, calculating True Positive Rate (TPR), False Positive Rate (FPR), and False Negative Rate (FNR) at each step.
  3. *Equal Error Rate (EER) Calculation*: Identifies the decision threshold operating point where $|FPR - FNR|$ is minimized (equalizing false alarm and miss rates).
  4. *FPR < 1% Conservative Threshold*: Computes the conservative decision operating point where False Positive Rate $FPR < 0.01$ (less than 1% false positive risk) to minimize false accusations against authentic human speakers.
  5. *4-Tier Threshold Calibration*: Establishes four calibrated score operating ranges:
     - **0 – 35%**: **Authentic Human Voice** (high-confidence genuine human speech)
     - **36 – 55%**: **Likely Human Voice** (moderate-confidence human speech)
     - **56 – 74%**: **Likely AI Voice** (moderate-confidence synthetic/deepfake speech, starting at the 56.0% decision threshold)
     - **75 – 100%**: **Authentic AI Voice** (high-confidence AI/TTS generated voice)
  6. *Results Persistence*: Saves raw per-clip metrics to `audio_forensics/calibration/calibration_results.json` and ROC curve data to `audio_forensics/calibration/roc_data.json`. Updates `audio_forensics/calibration/model_config.json` with `"calibration_status": "calibrated"`, `"decision_threshold_pct": 56.0`, and the 4-tier configuration.
* **What comes out**: JSON evaluation datasets (`calibration_results.json`, `roc_data.json`) and updated `model_config.json` threshold configuration.

**Key variables/objects worth knowing**:
* `roc_data`: List of dictionaries containing TPR, FPR, FNR, TP, FP, TN, and FN counts per 0.5% threshold step.
* `eer_row`: Dict containing the operating point metrics at the Equal Error Rate.
* `fpr1_row`: Dict containing the operating point metrics where FPR < 1%.
* `decision_threshold_pct`: The binary decision boundary set at 56.0% for AI voice detection.
* `threshold_tiers`: Dictionary mapping score ranges to the 4 verdict tiers (`authentic_human`: [0, 35], `likely_human`: [36, 55], `likely_ai`: [56, 74], `authentic_ai`: [75, 100]).

**Anything simplified/stubbed for now**:
Nothing is stubbed. ROC calculation, EER derivation, FPR < 1% search, 4-tier threshold calibration, and JSON persistence are 100% functional.

**How to verify this file works**:
Run `.\venv\Scripts\python.exe audio_forensics/calibration/run_calibration.py`. Inspect `audio_forensics/calibration/model_config.json` to verify updated calibration status and decision thresholds.

---

### File: `audio_forensics/calibration/model_config.json`

**What this file is responsible for**:
This is a version-controlled audit artifact that records both the verified label index mapping confirmed at model load time and the calibrated threshold parameters. It is read by `deepfake_model.py` and `fusion_integration.py` to drive verdict assignment and decision boundaries dynamically.

**Walkthrough of what happens, in order**:
* Generated automatically by `_load_model()` in `deepfake_model.py` on first model load and maintained by the calibration workflow.
* Contains: `model_name`, `id2label` (full map from model config), `fake_index` (int), `real_index` (int), `status` (`"verified"` once written by real code), `note` (explains derivation rule), and `calibration_parameters`.
* Under `calibration_parameters`:
  - `calibration_status`: `"calibrated"`
  - `calibration_note`: Calibration summary specifying the 4 verdict tiers.
  - `decision_threshold_pct`: `56.0` (decision boundary dividing Human Voice `<56%` and AI Voice `≥56%`).
  - `threshold_tiers`: Dictionary defining the exact range boundaries:
    * `authentic_human`: `[0, 35]` (Authentic Human Voice)
    * `likely_human`: `[36, 55]` (Likely Human Voice)
    * `likely_ai`: `[56, 74]` (Likely AI Voice)
    * `authentic_ai`: `[75, 100]` (Authentic AI Voice)
  - `temperature_scaling`: `1.15` (logit scaling factor before Sigmoid evaluation).

**Key variables/objects worth knowing**:
* `status`: Starts as `"unverified_stub"` in the repo; becomes `"verified"` the first time `_load_model()` completes successfully.
* `calibration_status`: Set to `"calibrated"` after threshold tuning.
* `threshold_tiers`: Specifies the 4 score tiers for voice authenticity classification.

**Anything simplified/stubbed for now**:
Nothing — the file is fully persisted and loaded dynamically at runtime.

**How to verify this file works**:
After running any test or pipeline analysis, open `audio_forensics/calibration/model_config.json` and confirm `status == "verified"`, `calibration_status == "calibrated"`, and `threshold_tiers` contains the four ranges [0-35, 36-55, 56-74, 75-100].

---

### File: `audio_forensics/tests/test_deepfake_model.py`

**What this file is responsible for**:
This test module validates `deepfake_model.py` across six distinct concerns: return schema, value types, score range, formula correctness, file path input, and calibration file integrity.

**Walkthrough of what happens, in order**:
* **What comes in**: PyTest triggers; most tests use a `np.zeros(16000)` dummy waveform so they run without external audio files.
* **Step-by-step logic**:
  1. *Schema test*: Asserts exactly the three required dict keys (`s_audio`, `logit_fake`, `logit_real`) are present.
  2. *Type test*: Asserts all three values are Python `float`.
  3. *Range test*: Asserts `0.0 ≤ s_audio ≤ 100.0`.
  4. *Formula test*: Independently recomputes `Sigmoid(logit_fake - logit_real) × 100` and asserts it matches `s_audio` to within 0.001 — catching any future formula drift.
  5. *Filepath test*: Conditionally calls `score_audio()` with `speech_sample.wav` if the file exists; skipped otherwise.
  6. *Calibration test*: Asserts `model_config.json` exists post-load with `status == "verified"` and integer indices.
  7. *Label derivation test*: Cross-checks that the `fake_index` and `real_index` values in `model_config.json` actually point to `'fake'` and `'real'` labels in `id2label`.
* **What comes out**: Test pass/fail assertions.

**Key variables/objects worth knowing**:
* `CALIBRATION_PATH`: Points to `calibration/model_config.json` relative to the test file.

**Anything simplified/stubbed for now**:
Fully functional. All six tests exercise real model outputs.

**How to verify this file works**:
Run `.\venv\Scripts\pytest.exe audio_forensics/tests/test_deepfake_model.py -v` for verbose per-test output.

---

## Phase 2.5 — Signal Log & Output Schema

### File: `audio_forensics/pipeline.py`

**What this file is responsible for**:
This file is the public entrypoint for the entire audio forensics subsystem. Its single job is to chain all three internal stages — VAD silence stripping, ASR transcription, and deepfake classification — in the correct order, then assemble the results into the contractually fixed output schema that `fusion.py` and the Cross-Modal Consistency Check consume. No external caller should ever need to import `vad.py`, `asr.py`, or `deepfake_model.py` directly.

**Walkthrough of what happens, in order**:
* **What comes in**: `audio_path`, a string filepath to an audio file (`.wav`, `.mp3`, etc.). Raises `FileNotFoundError` immediately if the path does not exist on disk.
* **Step-by-step logic**:
  1. *Existence Guard*: `os.path.exists(audio_path)` is checked before any processing begins. This surfaces bad paths with a clear error at the pipeline boundary rather than deep inside a model loader.
  2. *Stage 1 — VAD (`strip_silence`)*: Calls `vad.strip_silence(audio_path)` to detect speech timestamps, strip silence, and return a clean 16kHz float32 mono `numpy.ndarray` called `processed_audio`.
  3. *Stage 2 — ASR (`transcribe`)*: Passes `processed_audio` to `asr.transcribe()`, which runs Whisper-Tiny and returns a plain Python `str` containing the spoken text.
  4. *Stage 3 — Deepfake Scoring (`score_audio`)*: Passes the same `processed_audio` to `deepfake_model.score_audio()`, which runs wav2vec2 inference and returns `{"s_audio": float, "logit_fake": float, "logit_real": float}`.
  5. *Signal Logging*: Each stage emits a timestamped `[pipeline]` log line reporting the sample count, audio duration in seconds, character count of the transcript, model scores, and per-stage elapsed time. This makes the pipeline debuggable without a separate profiling tool.
  6. *Schema Assembly*: Merges all results into the fixed five-key dict. Key names are **contractually fixed** — `audio_score` and `transcript` are referenced by name in `fusion.py` and the Cross-Modal Consistency Check; renaming them will break those consumers silently.
* **What comes out**:
  ```json
  {
    "audio_score":         81.2,
    "logit_fake":          2.14,
    "logit_real":          -0.87,
    "transcript":          "text transcribed from audio...",
    "wer_confidence_note": "internal sanity-check only, not per-clip WER"
  }
  ```

**Key variables/objects worth knowing**:
* `processed_audio`: The shared 16kHz float32 mono waveform passed into both `transcribe()` and `score_audio()`. It is computed once from VAD and reused, so the two models always operate on the same signal.
* `audio_score`: Renamed from `s_audio` here to match the fusion contract (the internal deepfake model uses `s_audio`; the external pipeline exposes `audio_score`).
* `wer_confidence_note`: A fixed informational string for fusion consumers. No per-clip Word Error Rate is computed; this note exists to clarify that `wer_confidence_note` is a schema placeholder for a future evaluation feature, not a live accuracy metric.
* `time.perf_counter()`: Used for per-stage wall-clock timing in the log lines. Does not affect return values.

**Anything simplified/stubbed for now**:
Nothing is stubbed. All three stages call real models with real inference. The `wer_confidence_note` key carries a fixed string rather than a live WER metric — computing WER requires a ground-truth reference transcript, which is a Phase 3 / cross-modal concern.

**How to verify this file works**:
Run `.\venv\Scripts\pytest.exe audio_forensics/tests/test_audio_pipeline.py -v`. All five tests should pass. You can also run a quick manual smoke-test:
```
.\venv\Scripts\python.exe -c "from audio_forensics.pipeline import analyze_audio; import json; print(json.dumps(analyze_audio('audio_forensics/sample_clips/test_clip_with_silence.wav'), indent=2))"
```

---

### File: `audio_forensics/tests/test_audio_pipeline.py`

**What this file is responsible for**:
This test module validates `audio_forensics/pipeline.py` end-to-end across five distinct concerns: full schema presence, value type correctness, `audio_score` range, transcript type enforcement, the `FileNotFoundError` guard, and the fixed `wer_confidence_note` contract string.

**Walkthrough of what happens, in order**:
* **What comes in**: PyTest execution triggers; the `setup_module` fixture generates a synthetic WAV clip if one does not already exist.
* **Step-by-step logic**:
  1. *Full Pipeline Test (`test_analyze_audio_pipeline`)*: Calls `analyze_audio()` on the synthetic clip and asserts that all five keys (`audio_score`, `logit_fake`, `logit_real`, `transcript`, `wer_confidence_note`) are present with correct Python types.
  2. *Range Test (`test_audio_score_range`)*: Asserts `0.0 ≤ audio_score ≤ 100.0`, enforcing the Sigmoid × 100 mathematical guarantee.
  3. *Transcript Type Test (`test_transcript_is_string`)*: Asserts the transcript is a plain `str`, not a `dict` or `list`.
  4. *Missing File Test (`test_missing_file_raises_error`)*: Passes a nonexistent path and asserts that `FileNotFoundError` is raised, verifying the existence guard.
  5. *Contract String Test (`test_wer_confidence_note_is_fixed_string`)*: Asserts the `wer_confidence_note` value matches the exact fixed string, catching any accidental modification that would break downstream consumers who check this field.
* **What comes out**: Test pass/fail assertions.

**Key variables/objects worth knowing**:
* `SAMPLE_CLIP_PATH`: Points to `sample_clips/test_clip_with_silence.wav`, a synthetic 5-second clip (1.5s silence + 2s 440Hz tone + 1.5s silence) generated by `setup_module` if absent.

**Anything simplified/stubbed for now**:
Fully functional. All five tests exercise real end-to-end inference.

**How to verify this file works**:
Run `.\venv\Scripts\pytest.exe audio_forensics/tests/test_audio_pipeline.py -v` directly.

---

## Phase 2.6 — Testing & Validation

### File: `audio_forensics/tests/test_audio_pipeline.py`

**What this file is responsible for**:
This test module provides end-to-end integration testing for `audio_forensics/pipeline.py` across 8 distinct validation scenarios, including realistic spoken human/synthetic audio, pure tone non-speech clips, and heavy silence signals (88% silence).

**Walkthrough of what happens, in order**:
* **Step-by-step test execution**:
  1. `test_analyze_audio_pipeline`: Validates dict output schema, type assertions, and required key presence (`audio_score`, `logit_fake`, `logit_real`, `transcript`, `wer_confidence_note`).
  2. `test_audio_score_range`: Asserts `0.0 ≤ audio_score ≤ 100.0`.
  3. `test_transcript_is_string`: Verifies transcript return type is `str`.
  4. `test_missing_file_raises_error`: Asserts `FileNotFoundError` is raised for invalid paths.
  5. `test_wer_confidence_note_is_fixed_string`: Asserts exact contract string match.
  6. `test_human_speech_clip`: Runs pipeline on `speech_sample.wav` and verifies spoken transcript extraction.
  7. `test_synthetic_tone_clip`: Runs pipeline on pure tone clip and asserts VAD handles non-speech signals without crashing.
  8. `test_silence_heavy_clip`: Runs pipeline on 88% silence-heavy audio and verifies VAD silence stripping under heavy silence conditions.

---

### File: `audio_forensics/sample_clips/README.md`

**What this file is responsible for**:
Documents benchmark test results across audio clip scenarios and details the critical real-world generalization gap for `garystafford/wav2vec2-deepfake-voice-detector`.

**Key takeaways**:
* **Training set size**: The model was trained on ~1,866 samples. While self-reported AUC is 0.998, unseen voice cloners (ElevenLabs, Bark, XTTS v2, OpenAI Voice) present domain shift.
* **Audit protocol**: Requires a 20–30 clip manual audit on unseen TTS engines before production threshold tuning.

---

## Phase 2.7 — Calibration & Speed Optimisation

### Directory: `audio_forensics/calibration/`

**What this folder is responsible for**:
Handles empirical threshold calibration and configurations to map raw model logits to calibrated real-world confidence tiers. It acts as the centralized config storage for the audio detection parameters.

**Key components**:
* `model_config.json`: The central configuration mapping. Contains label mapping metadata (`fake_index`, `real_index`) verified dynamically at startup, and `calibration_parameters` including the calibrated decision threshold (`56.0%`), temperature scaling parameter (`1.15`), and boundaries for the 4-tier confidence system.
* `run_calibration.py`: A utility script that scores folders of authentic (real) and generated (fake) audio files, sweeps thresholds to compute ROC curves, identifies the Equal Error Rate (EER), and recommends optimal thresholds.
* `calibration_results.json` & `roc_data.json`: Persisted evaluation runs and statistics derived from the calibration protocol.

---

### File: `audio_forensics/deepfake_model.py` (Speed Optimizations)

**What this file is responsible for**:
Classifies audio segments as real or deepfake. In Phase 2.7, it was optimized to run and return outputs significantly faster (under 60 seconds) through:
1. **Offline-first model loading**: Attempts to initialize model and processor weights with `local_files_only=True` first, falling back to network-enabled loading only if the model is not locally cached. This bypasses slow network version checking on startup (saving ~60-70 seconds).
2. **Batched sliding-window scoring**: Instead of scoring windows sequentially in a loop, it batches all sliding windows (5s window, 1s overlap) into a single processor and model forward pass. This leverages parallel CPU/GPU vectorization for rapid execution.
3. **No Double Inference**: Returns logits for all windows directly from the initial batched inference, eliminating a second model pass on the worst window.

---

## Frontend UI — Streamlit Interactive Web Application

### File: `app.py` (Root) & `audio_forensics/app.py`

**What these files are responsible for**:
Provide the interactive tabbed web user interface to upload audio clips, run the VAD + ASR + Deepfake forensics pipeline, and display results.

**Key features & UI layout**:
1. **Multi-Format Uploader**: Accepts `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, and `.aac` formats, with an in-browser audio player.
2. **4-Tier Verdict Badge**: Shows the calibrated severity tier based on the deepfake risk score:
   - 🟢 **Authentic Human Voice** (Score: `0–35%`)
   - 🟢 **Likely Human Voice** (Score: `36–55%`)
   - 🟡 **Likely AI Voice** (Score: `56–74%`, starting at decision threshold `56.0%`)
   - 🔴 **Authentic AI Voice** (Score: `75–100%`)
3. **Speech-to-Text Transcript (Whisper-Tiny)**: Displays the transcribed spoken text.
4. **Scoring Margins & Logits Breakdown**: Displays metrics for Deepfake Score, Logit (Fake), Logit (Real), and Logit Difference Margin $\Delta$, alongside a synthetic risk progress bar and scoring formula details.
5. **Process Timing Breakdown**: Shows execution times for individual components:
   - **Silero VAD** duration (in seconds)
   - **Whisper ASR** duration (in seconds)
   - **Deepfake Score** duration (in seconds)
   - **Total Pipeline** execution time (in seconds)
6. **Analytical Classification Explanation**: Provides plain-language explanations of logit dominance, sigmoid scaling, confidence level, and underlying acoustic patterns.

---

## Code to Run the Streamlit POC

To launch the Streamlit Frontend POC web application, execute the following command in your terminal from the project root directory:

```bash
.\venv\Scripts\streamlit.exe run audio_forensics/app.py
```

Or using python:

```bash
.\venv\Scripts\python.exe -m streamlit run audio_forensics/app.py
```

Once executed, Streamlit will start a local server and automatically open the application in your browser at:
`http://localhost:8501`
