# Expanded Multi-Domain HC3 Text Forensics Corpus (v2.0)

## 1. Overview
This dataset expands the baseline HC3 corpus with balanced, verified passages across 18 distinct domains.
- **Total Samples**: 781
- **Human Authoring**: 394 samples (50.4%)
- **AI Generations**: 387 samples (49.6%)
- **Domain Count**: 18 balanced writing domains
- **Length Diversity**: Short (30-80w), Medium (80-250w), and Long (250-700w)

## 2. Integrity & Leakage Rules
1. **Zero LLM-Generated Human Data**: Human samples are sourced from legitimate public repositories (OpenStax, RFCs, SEC EDGAR, PubMed, MIT OCW, Wikipedia).
2. **Benchmark Protection**: Diagnostic evaluation benchmarks are strictly excluded.
3. **Multi-Model AI Distribution**: Generative representations span GPT-4, Claude 3.5, Gemini 1.5, LLaMA 3, and Mistral.
