# Phase 1.8 Validation Results — Corrected (HC3 Real Data)

> **Note:** The previous version of this file (before Correction 2) used 20
> author-written paragraphs imitating AI style — not real LLM output. Those
> numbers were not meaningful accuracy measures and are replaced entirely here.

## Test Set Composition
- Source: Hello-SimpleAI/HC3 (all.jsonl, held out from calibration)
- Human: 20 samples (human_answers field)
- AI: 20 samples (chatgpt_answers field)
- Total scored: 40

## Performance Metrics (threshold = 50)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9650 |
| F1-score (threshold=50) | 0.8333 |

## Confusion Matrix (threshold = 50)

| | Predicted Human | Predicted AI |
|---|---|---|
| **Actually Human** | TN=12 | FP=8 |
| **Actually AI** | FN=0 | TP=20 |

## Per-Sample Scores

| # | True Label | Score | Prediction | Correct |
|---|-----------|-------|-----------|---------|
| 1 | HUMAN | 63.8 | AI | ❌ |
| 2 | HUMAN | 55.1 | AI | ❌ |
| 3 | HUMAN | 52.4 | AI | ❌ |
| 4 | HUMAN | 46.1 | Human | ✅ |
| 5 | HUMAN | 74.6 | AI | ❌ |
| 6 | HUMAN | 28.8 | Human | ✅ |
| 7 | HUMAN | 45.8 | Human | ✅ |
| 8 | HUMAN | 57.0 | AI | ❌ |
| 9 | HUMAN | 51.1 | AI | ❌ |
| 10 | HUMAN | 47.1 | Human | ✅ |
| 11 | HUMAN | 43.3 | Human | ✅ |
| 12 | HUMAN | 20.6 | Human | ✅ |
| 13 | HUMAN | 39.4 | Human | ✅ |
| 14 | HUMAN | 50.2 | AI | ❌ |
| 15 | HUMAN | 29.7 | Human | ✅ |
| 16 | HUMAN | 19.2 | Human | ✅ |
| 17 | HUMAN | 49.5 | Human | ✅ |
| 18 | HUMAN | 62.1 | AI | ❌ |
| 19 | HUMAN | 30.3 | Human | ✅ |
| 20 | HUMAN | 45.1 | Human | ✅ |
| 21 | AI | 60.6 | AI | ✅ |
| 22 | AI | 68.1 | AI | ✅ |
| 23 | AI | 64.0 | AI | ✅ |
| 24 | AI | 75.3 | AI | ✅ |
| 25 | AI | 68.4 | AI | ✅ |
| 26 | AI | 72.3 | AI | ✅ |
| 27 | AI | 69.0 | AI | ✅ |
| 28 | AI | 68.8 | AI | ✅ |
| 29 | AI | 76.7 | AI | ✅ |
| 30 | AI | 73.4 | AI | ✅ |
| 31 | AI | 72.3 | AI | ✅ |
| 32 | AI | 75.6 | AI | ✅ |
| 33 | AI | 65.3 | AI | ✅ |
| 34 | AI | 84.5 | AI | ✅ |
| 35 | AI | 75.9 | AI | ✅ |
| 36 | AI | 83.1 | AI | ✅ |
| 37 | AI | 72.3 | AI | ✅ |
| 38 | AI | 71.7 | AI | ✅ |
| 39 | AI | 75.2 | AI | ✅ |
| 40 | AI | 82.0 | AI | ✅ |

## Edge Case Results

### Under 50 words
- text_score: 86.9
- stability_flag: stable
- curvature_raw: -0.8508601188659668
- burstiness_raw: None
- cliche_density_pct: 0.00%
- entropy: 3.808 bits

### Half human / half AI (spliced)
- text_score: 47.1
- stability_flag: stable
- curvature_raw: -1.5923891067504883
- burstiness_raw: None
- cliche_density_pct: 15.28%
- entropy: 5.844 bits

### HC3 AI text, lightly hand-edited
- text_score: 78.4
- stability_flag: stable
- curvature_raw: -0.4685027599334717
- burstiness_raw: None
- cliche_density_pct: 0.00%
- entropy: 5.724 bits

## Errors
- 40 sample(s) failed to score.
  - Sample 1 (HUMAN): 'charmap' codec can't encode character '\u274c' in position 25: character maps to <undefined>
  - Sample 2 (HUMAN): 'charmap' codec can't encode character '\u274c' in position 25: character maps to <undefined>
  - Sample 3 (HUMAN): 'charmap' codec can't encode character '\u274c' in position 25: character maps to <undefined>
  - Sample 4 (HUMAN): 'charmap' codec can't encode character '\u2705' in position 28: character maps to <undefined>
  - Sample 5 (HUMAN): 'charmap' codec can't encode character '\u274c' in position 25: character maps to <undefined>
  - Sample 6 (HUMAN): 'charmap' codec can't encode character '\u2705' in position 28: character maps to <undefined>
  - Sample 7 (HUMAN): 'charmap' codec can't encode character '\u2705' in position 28: character maps to <undefined>
  - Sample 8 (HUMAN): 'charmap' codec can't encode character '\u274c' in position 25: character maps to <undefined>
  - Sample 9 (HUMAN): 'charmap' codec can't encode character '\u274c' in position 25: character maps to <undefined>
  - Sample 10 (HUMAN): 'charmap' codec can't encode character '\u2705' in position 28: character maps to <undefined>
  - Sample 11 (HUMAN): 'charmap' codec can't encode character '\u2705' in position 28: character maps to <undefined>
  - Sample 12 (HUMAN): 'charmap' codec can't encode character '\u2705' in position 28: character maps to <undefined>
  - Sample 13 (HUMAN): 'charmap' codec can't encode character '\u2705' in position 28: character maps to <undefined>
  - Sample 14 (HUMAN): 'charmap' codec can't encode character '\u274c' in position 25: character maps to <undefined>
  - Sample 15 (HUMAN): 'charmap' codec can't encode character '\u2705' in position 28: character maps to <undefined>
  - Sample 16 (HUMAN): 'charmap' codec can't encode character '\u2705' in position 28: character maps to <undefined>
  - Sample 17 (HUMAN): 'charmap' codec can't encode character '\u2705' in position 28: character maps to <undefined>
  - Sample 18 (HUMAN): 'charmap' codec can't encode character '\u274c' in position 25: character maps to <undefined>
  - Sample 19 (HUMAN): 'charmap' codec can't encode character '\u2705' in position 28: character maps to <undefined>
  - Sample 20 (HUMAN): 'charmap' codec can't encode character '\u2705' in position 28: character maps to <undefined>
  - Sample 21 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 22 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 23 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 24 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 25 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 26 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 27 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 28 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 29 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 30 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 31 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 32 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 33 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 34 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 35 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 36 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 37 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 38 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 39 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>
  - Sample 40 (AI): 'charmap' codec can't encode character '\u2705' in position 25: character maps to <undefined>

## Known Limitations
- Calibration baseline uses distilgpt2, which shows weak discrimination (Correction 3).
- distilgpt2 curvature values overlap between human and AI text.
- HC3 AI answers are ChatGPT-3.5 answers (not GPT-4 or other models).
- Test set is small (20+20); results should be treated as indicative.
- Honest assessment: burstiness and entropy carry more signal than curvature at this model scale.