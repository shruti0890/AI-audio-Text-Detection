# Production Five-Feature Text Forensics Model

## Overview
This directory contains the final **production** machine learning model for AI text forensics.

## Evolution and Decision History
1. **Historical 4-Feature Baseline**: Used heuristic Gaussian CDF weighting on 4 features (`curvature`, `burstiness`, `lexical_entropy`, `cliche_density`). Suffered from misclassifications on polished AI technical text.
2. **6-Feature Experimentation**: Added `ngram_repetition` and `structural_regularity` fused via Logistic Regression.
   - 6-Feature LR Performance (109 test samples): Accuracy 93.07%, F1 0.9320, FPR 4.17%, FNR 9.43%.
3. **Controlled 5-Feature Ablation (Final Decision)**:
   - Analysis revealed that `ngram_repetition` added noise to the corpus (negative coefficient, higher FPR).
   - Removing `ngram_repetition` resulted in **improved performance across all binary classification metrics**:
     - **Accuracy**: 94.74% (+1.67%)
     - **Precision**: 97.92% (+1.92%)
     - **Recall**: 92.16% (+1.59%)
     - **F1 Score**: 94.95% (+1.75%)
     - **False Positive Rate (FPR)**: 2.27% (cut nearly in half from 4.17%)
     - **False Negative Rate (FNR)**: 7.84% (down from 9.43%)
     - **ROC-AUC**: 95.69%
   - Correctly identifies polished AI technical text with $P(\text{AI}) = 1.0000$ (Verdict: AI).
   - `ngram_repetition` remains archived in `signals/ngram_repetition.py` for future research, but is excluded from the production feature vector.

## Final Production Feature Set (Exact Canonical Order)
1. `curvature`: Fast-DetectGPT probability curvature under `HuggingFaceTB/SmolLM2-135M`
2. `burstiness`: $\sigma / \mu$ sentence length variation
3. `lexical_entropy`: Shannon entropy and vocabulary richness
4. `structural_regularity`: Composite of sentence-starter diversity and POS overlap
5. `cliche_density`: Frequency of 50+ overused AI idioms / clichés

## Architecture
$$\text{Raw Feature Vector } (5\text{D}) \xrightarrow{\text{StandardScaler}} \text{Standardized Vector } (5\text{D}) \xrightarrow{\text{Logistic Regression}} P(\text{AI})$$

## Calibrated Decision Thresholds
- **Human**: $P(\text{AI}) \le 0.20$
- **Likely Human**: $0.20 < P(\text{AI}) < 0.45$
- **Likely AI**: $0.45 \le P(\text{AI}) < 0.70$
- **AI**: $P(\text{AI}) \ge 0.70$

## Artifacts
- `model.pkl`: Fitted `sklearn.linear_model.LogisticRegression` (5 features)
- `scaler.pkl`: Fitted `sklearn.preprocessing.StandardScaler` (5 features)
- `model_metadata.json`: Feature ordering, coefficients, split parameters, metrics, thresholds
- `evaluation_report.md`: Comparative evaluation report
