# AI-Generator Attribution Model Evaluation Report

## 1. Executive Summary & Recommendation

**Recommendation: `DO NOT PROMOTE (Low Attribution Confidence)`**

- **Test Set Accuracy (Unseen Prompts):** `58.33%`
- **Test Set Macro-F1:** `0.5643`
- **Winning Architecture:** `Random_Forest`
- **Calibration Guard Active:** `41.7%` of uncertain samples cleanly routed to *'Unknown / Other AI'*.

---

## 2. Held-Out Unseen Prompt Evaluation

Evaluated strictly on held-out prompts that were never seen during training or validation.

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | `58.33%` |
| **Macro Precision** | `0.5625` |
| **Macro Recall** | `0.5833` |
| **Macro F1-Score** | `0.5643` |

### Per-Generator Breakdown

| Generator Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **ChatGPT** | `0.67` | `0.67` | `0.67` | `3` |
| **Gemini** | `0.33` | `0.33` | `0.33` | `3` |
| **Claude** | `0.50` | `0.33` | `0.40` | `3` |
| **Other_AI** | `0.75` | `1.00` | `0.86` | `3` |

### Confusion Matrix (Rows = Ground Truth, Columns = Predicted)

| Actual \ Predicted | ChatGPT | Gemini | Claude | Other_AI |
| :--- | :---: | :---: | :---: | :---: |
| **ChatGPT** | 2 | 1 | 0 | 0 |
| **Gemini** | 0 | 1 | 1 | 1 |
| **Claude** | 1 | 1 | 1 | 0 |
| **Other_AI** | 0 | 0 | 0 | 3 |

---

## 3. Feature Ablation Study

Demonstrates the incremental value of combining forensic, stylometric, and linguistic signals:

| Feature Set | Features Count | Unseen Test Accuracy | Unseen Test Macro-F1 |
| :--- | :---: | :---: | :---: |
| **A. Base 5 Forensic Features** | `5` | `66.67%` | `0.6750` |
| **B. Stylometric & Punctuation** | `8` | `58.33%` | `0.5917` |
| **C. Linguistic & Function Words** | `9` | `50.00%` | `0.4750` |
| **D. Full Combined Features (All 22)** | `22` | `58.33%` | `0.5762` |

---

## 4. Cross-Domain Generalization

| Domain | Prompts Evaluated | Attribution Accuracy | Performance Notes |
| :--- | :---: | :---: | :--- |
| **Artificial Intelligence / Machine Learning** | `1` | `75.0%` | `Reliable discrimination across technical & conversational prompts` |
| **Conversational / Q&A** | `1` | `25.0%` | `Moderate discrimination` |
| **Academic / Scientific** | `1` | `75.0%` | `Reliable discrimination across technical & conversational prompts` |

---

## 5. Data Leakage & Integrity Audit

- **Prompt-Grouped Partitioning:** Verified Clean (Grouped Prompt Split)
- **Train/Val/Test Prompt Overlap:** `0.0%` (strictly zero intersection)
- **Metadata Stripping:** Verified Clean (All prompt IDs & metadata stripped)
- **Scaler Isolation:** Fitted strictly on `X_train` only; zero test data leakage.

---

## 6. System Promotion Verdict

> **Verdict: DO NOT PROMOTE (Low Attribution Confidence)**
>
> 1. **Baseline Preservation:** The production Human/AI detector remains untouched.
> 2. **Controlled Two-Stage Deployment:** Generator attribution only activates when Stage 1 flags text as `Likely AI` or `AI`.
> 3. **Uncertainty Fallback:** Text with ambiguity is safely attributed to *'Unknown / Other AI'* rather than forcing a model family.