# AUDIO_CALIBRATION_PROTOCOL.md
## Empirical Threshold Calibration for `wav2vec2-deepfake-voice-detector`

**Status**: Pending execution — current `decision_threshold_pct: 62.5` is a placeholder  
**Analogous to**: Text Correction 9 (grid-search that produced ROC-AUC 0.9994 on HC3)  
**Target**: Find the threshold where False Positive Rate (FPR) < 1% on modern TTS engines  
**When done**: Update `audio_forensics/calibration/model_config.json` — **no code changes needed**

---

## Why This Is Needed

The `garystafford/wav2vec2-deepfake-voice-detector` was pre-trained on ~1,866 clips from the
**ASVspoof 2019** dataset. Modern TTS engines (ElevenLabs, Bark, XTTS v2, OpenAI TTS) were
not in the training set — they produce cleaner, higher-fidelity speech that causes **domain
shift**: the model's logit distributions compress or shift, making the naive 50% sigmoid
boundary unreliable.

The current `62.5%` placeholder was derived from the published ASVspoof 2021 EER for
wav2vec2-class models. The goal of this protocol is to **replace that estimate with a
number derived from your own test clips**.

---

## Step 1 — Collect Test Clips (30–50 per class)

### 1A. Fake / Synthetic clips → `audio_forensics/sample_clips/fake/`

| Engine | Access | Target Clips |
|---|---|---|
| **ElevenLabs** | https://elevenlabs.io (free tier: 10k chars/month) | 10–15 clips |
| **OpenAI TTS** | `openai.audio.speech.create()` via API | 5–10 clips |
| **Bark** | `pip install bark` (local, free) | 5–10 clips |
| **XTTS v2** | `pip install TTS` + `tts_models/multilingual/multi-dataset/xtts_v2` | 5–10 clips |
| **Coqui TTS** | `pip install TTS` — any English model | 5–10 clips |

**Suggested test sentence** (covers broad phoneme range):
> "The quick brown fox jumps over the lazy dog near the riverbank on a sunny afternoon. This sentence contains a wide variety of phonemes and is commonly used in speech testing."

**Format**: WAV, any sample rate (pipeline resamples to 16kHz), mono preferred, 5–15 seconds per clip.

### 1B. Real / Human clips → `audio_forensics/sample_clips/real/`

| Source | Access | Notes |
|---|---|---|
| **Your own voice** | Phone/mic recording | Best ground truth |
| **Common Voice (Mozilla)** | https://commonvoice.mozilla.org/en/datasets | Public domain, validated |
| **VCTK Corpus** | https://datashare.ed.ac.uk/handle/10283/2651 | 110 English speakers |
| **LibriSpeech test-clean** | https://www.openslr.org/12 | Studio-quality |

---

## Step 2 — Run the Calibration Script

```bash
# From the repository root (Integration_branch)
python audio_forensics/calibration/run_calibration.py
```

The script (`run_calibration.py`) will:
1. Score all clips in `fake/` and `real/` subdirectories
2. Sweep decision thresholds from 0% to 100% in 0.5% steps
3. Print the EER threshold (FPR = FNR)
4. Print the FPR < 1% threshold (most conservative operating point)
5. Save `calibration_results.json` and `roc_data.json`

---

## Step 3 — Interpret Output

### What Good Output Looks Like (well-separated distributions)

```
FAKE  elevenlabs_clip1.wav: 89.4%
FAKE  bark_sample.wav:      78.2%
REAL  my_voice.wav:         21.3%
REAL  commonvoice_001.wav:  18.7%

=== EER ===
Threshold: 55.0%  |  EER: 0.041  |  FPR=0.042  FNR=0.040

=== FPR < 1% Threshold ===
Threshold: 71.5%  |  TPR=0.8800  FPR=0.0083  |  TP=44  FP=0  TN=48  FN=6

=== RECOMMENDED UPDATE ===
Set decision_threshold_pct = 71.5
```

### Metric Definitions

| Metric | Definition | Target |
|---|---|---|
| **EER** | Equal Error Rate — where FPR = FNR | As low as possible |
| **FPR** | False Positive Rate — real clips wrongly flagged as fake | **< 1%** |
| **TPR** | True Positive Rate — fake clips correctly caught | **> 80%** |
| **EER value** | (FPR + FNR) / 2 at EER threshold | **< 0.10** |

### What Bad Output Looks Like (overlapping distributions)
```
FAKE  elevenlabs_clip1.wav: 53.1%    ← barely above 50%
REAL  my_voice.wav:         49.7%    ← very close to fake score
→ Distributions overlap — model separates poorly on your clips.
→ Consider trying gpt2-medium equivalent (larger wav2vec2 model).
```

---

## Step 4 — Update `model_config.json`

After calibration, edit `audio_forensics/calibration/model_config.json`:

```json
{
  "calibration_parameters": {
    "calibration_status": "calibrated",
    "calibration_date": "YYYY-MM-DD",
    "calibration_dataset": "30 ElevenLabs/Bark/XTTS clips + 30 CommonVoice clips",
    "decision_threshold_pct": 71.5,
    "temperature_scaling": 1.15,
    "asvspoof_eer": 0.038,
    "empirical_eer": 0.041,
    "empirical_fpr_at_threshold": 0.0083,
    "empirical_tpr_at_threshold": 0.8800,
    "window_seconds": 5,
    "window_overlap_seconds": 1,
    "aggregation_strategy": "max_risk",
    "confidence_tiers": {
      "extreme_logit_margin": 3.0,
      "high_logit_margin": 1.5,
      "moderate_logit_margin": 0.5
    }
  }
}
```

Change `calibration_status` from `"placeholder"` to `"calibrated"`.  
**Restart the app** — no code changes needed. The pipeline loads from this file at startup.

---

## Step 5 — Validate on Held-Out Clips

Before declaring calibration final, test on **5–10 clips NOT used in Step 1**:

```bash
python -c "
import sys; sys.path.insert(0, '.')
from audio_forensics.pipeline import analyze_audio
clips = [
    ('audio_forensics/sample_clips/real/held_out_1.wav', 'REAL'),
    ('audio_forensics/sample_clips/fake/held_out_fake_1.wav', 'FAKE'),
]
for path, label in clips:
    r = analyze_audio(path)
    verdict = 'FAKE' if r['audio_score'] >= r['decision_threshold_pct'] else 'REAL'
    correct = '✅' if verdict == label else '❌'
    print(f'{correct} {label} → score={r[\"audio_score\"]:.1f}%  threshold={r[\"decision_threshold_pct\"]}%  verdict={verdict}')
"
```

---

## Acceptance Criteria

| Criterion | Required |
|---|---|
| FPR on held-out real clips | **< 1%** (at most 1 false positive per 100 real clips) |
| TPR on held-out fake clips | **> 80%** (catch at least 4 out of 5 deepfakes) |
| EER | **< 0.10** |
| `calibration_status` in JSON | **`"calibrated"`** |

---

## Score Distribution Visualization (Optional)

After generating `calibration_results.json`:

```python
# Run from repo root: python audio_forensics/calibration/plot_distribution.py
import json, numpy as np, matplotlib.pyplot as plt

with open("audio_forensics/calibration/calibration_results.json") as f:
    results = json.load(f)

real_scores = [r["score"] for r in results if r["label"] == 0]
fake_scores = [r["score"] for r in results if r["label"] == 1]

plt.figure(figsize=(10, 5))
plt.hist(real_scores, bins=20, alpha=0.6, color="green", label="Real Human Voice")
plt.hist(fake_scores, bins=20, alpha=0.6, color="red", label="Synthetic / Deepfake")
plt.axvline(x=62.5, color="orange", linestyle="--", label="Current threshold (placeholder 62.5%)")
plt.xlabel("Audio Score (%)"); plt.ylabel("Count")
plt.title("Score Distribution: Real vs Fake Audio Clips")
plt.legend(); plt.tight_layout()
plt.savefig("audio_forensics/calibration/score_distribution.png")
plt.show()
print("Saved: audio_forensics/calibration/score_distribution.png")
```

A well-calibrated model shows two clearly separated histogram peaks with minimal overlap at the threshold line.

---

## Analogy: Audio Calibration vs Text Correction 9

| | Text (Correction 9) | Audio (This Protocol) |
|---|---|---|
| Dataset | HC3 120 held-out samples | 30–50 real + 30–50 TTS clips |
| Method | Grid-search over fusion weights | ROC sweep, FPR < 1% criterion |
| Output | `fusion_config.json` weights | `model_config.json` threshold |
| Restart needed? | ✅ No code changes | ✅ No code changes |
| Result achieved | ROC-AUC 0.9994 | TBD after calibration |
| Status marker | N/A (weights replaced directly) | `calibration_status: "calibrated"` |
