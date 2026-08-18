# Five-Feature Ablation Experiment (Without ngram_repetition)

Generated: 2026-08-18 12:28:06

Dataset:   IDENTICAL to six-feature model: genre_corpus_additions_human/ai.json + hc3_test_pool_human/ai.json. Same seed, same split.
Split:     496 train / 106 val / 109 test
Features:  curvature, burstiness, lexical_entropy, structural_regularity, cliche_density
Excluded:  ngram_repetition

## Comparison: Five-Feature vs Six-Feature (Test Set)

| Model                                   | Accuracy | Precision | Recall |   F1  | ROC-AUC |  FPR |  FNR |
|-----------------------------------------|----------|-----------|--------|-------|---------|------|------|
| Five-Feature LR (without ngram_repetition) |   0.9474 |    0.9792 | 0.9216 | 0.9495 |  0.9569 | 0.0227 | 0.0784 |
| Six-Feature LR (reference)               |   0.9307 |      0.96 | 0.9057 |  0.932 |   0.962 | 0.0417 | 0.0943 |

### Delta (Five-Feature minus Six-Feature)

| Metric   | Delta |
|----------|-------|
| accuracy  | +0.0167 |
| precision | +0.0192 |
| recall    | +0.0159 |
| f1        | +0.0175 |
| roc_auc   | -0.0051 |
| fpr       | -0.0190 |
| fnr       | -0.0159 |

## Confusion Matrix (Five-Feature, Test Set)

```
                 Predicted Human  Predicted AI
Actual Human            43              1
Actual AI                4             47
```

n_definite = 95 / 109 total  (remaining 14 in uncertain zone)

## Calibrated Thresholds (Five-Feature LR)

- Human:        P(AI) ≤ 0.20
- Likely Human: 0.20 < P(AI) < 0.45
- Likely AI:    0.45 ≤ P(AI) < 0.70
- AI:           P(AI) ≥ 0.70

## Learned Coefficients (Five-Feature LR)

| Feature | Coefficient | Direction |
|---------|-------------|-----------|

| curvature | +3.9473 | → AI-like (higher) |
| burstiness | -1.1283 | → Human-like (higher) |
| lexical_entropy | -1.0621 | → Human-like (higher) |
| structural_regularity | -0.0372 | → Human-like (higher) |
| cliche_density | +0.8909 | → AI-like (higher) |

> Note: Signs after StandardScaler standardization. Positive = feature above its mean increases P(AI).

## Problematic AI Technical Paragraph — Regression Test

**P(AI) — Six-Feature model:**  ~1.0000 (AI)
**P(AI) — Five-Feature model:**  1.0
**Verdict — Five-Feature:**      AI

Feature values on problematic paragraph (5 features):
  - curvature: -0.5231
  - burstiness: 0.2921
  - lexical_entropy: 6.0951
  - structural_regularity: 27.2727
  - cliche_density: 2.2989

## Notes

- The six-feature model files (model.pkl, scaler.pkl, metadata) were not read or modified.
- This experiment uses the same random seed, split ratios, and feature cache as the six-feature model.
- All differences in results are attributable solely to the exclusion of ngram_repetition.
