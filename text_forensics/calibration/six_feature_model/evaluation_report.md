# Text Forensics — Old vs Corrected Baseline vs New Six-Feature Model

Generated: 2026-08-18 11:56:46

Dataset: genre_corpus_additions_human/ai.json + hc3_test_pool_human/ai.json. Multi-genre: technical, academic, news, legal, business, conversational, informational, creative. Multiple LLM sources included in AI corpus.

Split: 496 train / 106 val / 109 test

## Overall Metrics (Test Set)

| Model                                     | Accuracy | Precision | Recall |   F1  | ROC-AUC |  FPR |  FNR |
|-------------------------------------------|----------|-----------|--------|-------|---------|------|------|
| Old model (0.80/0.15/0.02/0.03)            |      1.0 |       1.0 |    1.0 |    1.0 |  0.9694 |   0.0 |   0.0 |
| Corrected baseline (0.65/0.20/0.05/0.10)   |      1.0 |       1.0 |    1.0 |    1.0 |   0.971 |   0.0 |   0.0 |
| New (six-feature LR)                       |   0.9307 |      0.96 | 0.9057 |  0.932 |   0.962 | 0.0417 | 0.0943 |

## Learned Feature Coefficients (Six-Feature LR)

| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| curvature | +4.0628 | → AI-like (higher) |
| burstiness | -0.7481 | → Human-like (higher) |
| lexical_entropy | -1.4540 | → Human-like (higher) |
| ngram_repetition | -0.8091 | → Human-like (higher) |
| structural_regularity | +0.1949 | → AI-like (higher) |
| cliche_density | +0.8511 | → AI-like (higher) |

> Note: Signs indicate direction AFTER standardization. Positive = feature value above mean increases AI probability.

## Calibrated Thresholds (Six-Feature LR)

- Human:       P(AI) ≤ 0.20
- Likely Human: 0.20 < P(AI) < 0.45
- Likely AI:   0.45 ≤ P(AI) < 0.70
- AI:          P(AI) ≥ 0.70

## Ablation Study (Test Set)

| Configuration | F1 | ROC-AUC | FPR |
|---------------|----|---------|-----|
| All 6 features | 0.932 | 0.962 | 0.0417 |
| Without curvature | 0.9259 | 0.8646 | 0.0952 |
| Without burstiness | 0.9333 | 0.9589 | 0.0426 |
| Without lexical_entropy | 0.9293 | 0.9441 | 0.0714 |
| Without ngram_repetition | 0.9495 | 0.9569 | 0.0227 |
| Without structural_regularity | 0.932 | 0.964 | 0.0612 |
| Without cliche_density | 0.9184 | 0.9579 | 0.0652 |

## Feature Correlations (Training Set)

| Feature Pair | Pearson r |
|--------------|-----------|
| curvature_vs_burstiness | -0.1692 |
| curvature_vs_lexical_entropy | -0.1595 |
| curvature_vs_ngram_repetition | 0.1988 |
| curvature_vs_structural_regularity | 0.2576 |
| curvature_vs_cliche_density | 0.181 |
| burstiness_vs_lexical_entropy | -0.0678 |
| burstiness_vs_ngram_repetition | 0.3901 |
| burstiness_vs_structural_regularity | -0.2291 |
| burstiness_vs_cliche_density | -0.0403 |
| lexical_entropy_vs_ngram_repetition | -0.3874 |
| lexical_entropy_vs_structural_regularity | 0.2603 |
| lexical_entropy_vs_cliche_density | -0.0256 |
| ngram_repetition_vs_structural_regularity | 0.0777 |
| ngram_repetition_vs_cliche_density | -0.0759 |
| structural_regularity_vs_cliche_density | 0.0467 |

## Problematic AI Technical Paragraph — Regression Test

**Old model AI probability:**  ~0.45 (text_score ~44.6 → prob ~0.45)
**Old model verdict:**         Human (misclassified)
**Baseline AI probability:**   ~0.45-0.55 (depends on weights)
**Baseline verdict:**          Likely Human
**New model AI probability:**  1.0
**New model verdict:**         AI

Feature values on problematic paragraph:
  - curvature: -0.5231
  - burstiness: 0.2921
  - lexical_entropy: 6.0951
  - ngram_repetition: 0.0
  - structural_regularity: 27.2727
  - cliche_density: 2.2989

## Limitations

- N-gram repetition and structural regularity are new features calibrated on this dataset.
- Short texts (< 30 words) are excluded from training.
- The system estimates AI probability; it does not definitively prove authorship.
- LLM output diversity continues to increase — periodic retraining is recommended.
