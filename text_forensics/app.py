"""
text_forensics/app.py

Streamlit Application for Text Forensics Deepfake Detection.
Production Five-Feature Logistic Regression Architecture.

Features:
  - Fused AI Probability & 4-Way Calibrated Verdict (Human / Likely Human / Likely AI / AI)
  - 5-Feature Forensic Breakdown (Curvature, Burstiness, Lexical Entropy, Structural Regularity, Cliché Density)
  - Sentence-Level QuillBot-Style Highlighting & AI Evidence Summary
  - Signal Disagreement Alert Banner
  - Cliché & Buzzword Word-Level Highlighting
  - Adversarial Robustness Check (T5 Paraphrase)
  - Raw Evidence Logs

Usage:
    streamlit run text_forensics/app.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import nltk
import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _APP_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from text_forensics.pipeline import analyze_text, _load_baseline_stats
from text_forensics.signals.cliche_scanner import CLICHE_TERMS
from text_forensics.signals.sentence_scorer import score_sentences

# Ensure nltk sent_tokenize data is ready
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Text Forensics Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .disclaimer-banner {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 1rem;
    }
    .cliche-highlight {
        background-color: #FECACA;
        color: #991B1B;
        font-weight: 600;
        padding: 0.15rem 0.35rem;
        border-radius: 0.25rem;
        border: 1px solid #FCA5A5;
    }
    .text-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 0.5rem;
        padding: 1.2rem;
        font-family: sans-serif;
        line-height: 1.8;
        font-size: 1.05rem;
        color: #1E293B;
    }
    .sent-high-ai {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 500;
        padding: 0.15rem 0.35rem;
        margin: 0 0.1rem;
        border-radius: 0.25rem;
        border-bottom: 2px solid #EF4444;
    }
    .sent-mid-ai {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.15rem 0.35rem;
        margin: 0 0.1rem;
        border-radius: 0.25rem;
        border-bottom: 2px solid #F59E0B;
    }
    .sent-low-ai {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.15rem 0.35rem;
        margin: 0 0.1rem;
        border-radius: 0.25rem;
        border-bottom: 2px solid #22C55E;
    }
    .sent-short {
        background-color: #F1F5F9;
        color: #64748B;
        padding: 0.15rem 0.35rem;
        margin: 0 0.1rem;
        border-radius: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar & Explanations
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 Text Forensics Engine")
    st.markdown(
        """
        **How Detection Works**
        
        The engine uses **5 independent forensic features** fused by a calibrated Logistic Regression model:
        
        1. 📈 **Curvature** — Fast-DetectGPT: log-probability discrepancy under `distilgpt2`.
        2. ⚡ **Burstiness** — σ/μ sentence-length variation: human writing naturally varies rhythm.
        3. 🔤 **Lexical Entropy** — Shannon entropy + TTR: AI text often exhibits uniform vocabulary.
        4. 🏗️ **Structural Regularity** — Sentence-starter diversity and POS pattern consistency.
        5. 🚩 **Cliché Density** — Frequency of 50+ overused AI buzzwords (*"rapidly evolving"*, *"delve into"*, *"pivotal"*).
        
        ---
        **Classification Thresholds (P(AI))**:
        - **Human**: $\le 0.20$
        - **Likely Human**: $0.20 < P < 0.45$
        - **Likely AI**: $0.45 \le P < 0.70$
        - **AI**: $\ge 0.70$
        
        ---
        **Robustness Check**
        Uses a neural T5 paraphrase model to test if score shifts under rephrasing.
        """
    )
    st.divider()
    st.caption("CPU-only | PyTorch | Transformers | NLTK | spaCy | scikit-learn")


# ---------------------------------------------------------------------------
# Header & Disclaimers
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">Text Forensics Deepfake Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Statistical forensic analysis to estimate AI vs human text authorship characteristics</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer-banner">'
    'ℹ️ <b>Forensic Assessment Note:</b> This system provides probabilistic estimates based on statistical text regularities. '
    'Results indicate whether text exhibits characteristics typical of AI language models, not definitive proof of authorship.'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Step 1: Input Section
# ---------------------------------------------------------------------------
st.subheader("1. Input Text")

input_mode = st.radio(
    "Choose Input Method:",
    options=["Paste Text", "Upload .txt File"],
    horizontal=True,
)

raw_text = ""

if input_mode == "Paste Text":
    raw_text = st.text_area(
        "Paste text to analyze:",
        height=220,
        placeholder="Paste plain text here...",
    )
else:
    uploaded_file = st.file_uploader("Upload a plain .txt file:", type=["txt"])
    if uploaded_file is not None:
        try:
            raw_text = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            raw_text = uploaded_file.read().decode("latin-1")


word_count = len(raw_text.split()) if raw_text else 0

if raw_text:
    st.caption(f"Input statistics: {word_count} words | {len(raw_text)} characters")
    if word_count < 30:
        st.warning(
            "⚠️ **Short Input Notice**: Text has under ~30 words. "
            "Sentence-length burstiness and structural signals require multiple sentences for full reliability."
        )

run_robustness = st.checkbox("Run T5 paraphrase robustness check (adds ~30s)", value=False)

analyze_btn = st.button(
    "Analyze Text",
    type="primary",
    disabled=not bool(raw_text and raw_text.strip()),
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# Step 2: Pipeline Execution & Display
# ---------------------------------------------------------------------------
if analyze_btn:
    if not raw_text or not raw_text.strip():
        st.error("Please provide valid, non-empty text input before analyzing.")
    else:
        st.divider()
        st.subheader("2. Forensic Analysis Results")

        t_start = time.time()

        try:
            spinner_msg = "Evaluating statistical forensic features, scoring sentences" + (" and running T5 robustness check..." if run_robustness else "...")
            with st.spinner(spinner_msg):
                result = analyze_text(raw_text, run_robustness=run_robustness)
                baseline = _load_baseline_stats()
                sentence_analysis = score_sentences(raw_text, baseline)
            elapsed = time.time() - t_start
        except Exception as e:
            st.error(f"❌ **Pipeline Error**: `{type(e).__name__}: {e}`")
            st.stop()

        # --- Signal Disagreement Banner ---
        if result.get("signal_agreement") == "disagreement":
            st.error(
                "⚠️ **Signals Disagree Significantly**: Individual feature sub-scores show spread > 40 points. "
                "Review the feature breakdown below carefully."
            )

        # --- Primary Metric Display ---
        ai_prob = result.get("ai_probability")
        lr_verdict = result.get("verdict", "Unknown")
        lr_confidence = result.get("confidence", "")
        model_used = result.get("model_used", "five_feature_logistic_regression")
        text_score = result.get("text_score", 50.0)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(
                label="Estimated AI Probability",
                value=f"{ai_prob * 100:.1f}%" if ai_prob is not None else "N/A",
                delta=f"{lr_verdict} ({lr_confidence} Confidence)",
                delta_color="inverse" if (ai_prob or 0) >= 0.45 else "normal",
            )
        with metric_col2:
            model_display_name = "Five-Feature Logistic Regression" if "five_feature" in model_used else "Corrected Baseline Fallback"
            st.metric(
                label="Classifier Model",
                value=model_display_name,
                delta="Calibrated on 711 multi-genre samples",
                delta_color="off",
            )
        with metric_col3:
            st.metric(
                label="Baseline Score (PATH A Reference)",
                value=f"{text_score:.1f} / 100",
                delta="Gaussian CDF Heuristic",
                delta_color="off",
            )

        st.divider()

        # --- QuillBot-Style Sentence-Level Highlighting ---
        st.subheader("3. Sentence-Level Breakdown")

        html_spans = []
        for s_item in sentence_analysis:
            s_text = s_item["sentence"]
            c_score = s_item.get("curvature_score")

            if c_score is None:
                span_html = f'<span class="sent-short" title="Insufficient context (<6 words)">{s_text}</span>'
            elif c_score >= 70:
                span_html = f'<span class="sent-high-ai" title="Sentence Curvature AI Score: {c_score:.1f}/100 (Strong AI)">{s_text}</span>'
            elif c_score >= 45:
                span_html = f'<span class="sent-mid-ai" title="Sentence Curvature AI Score: {c_score:.1f}/100 (Moderate AI)">{s_text}</span>'
            else:
                span_html = f'<span class="sent-low-ai" title="Sentence Curvature AI Score: {c_score:.1f}/100 (Human-like)">{s_text}</span>'

            html_spans.append(span_html)

        full_highlighted_doc = " ".join(html_spans)
        st.markdown(f'<div class="text-box">{full_highlighted_doc}</div>', unsafe_allow_html=True)

        sent_ev = result.get("sentence_evidence", {})
        if sent_ev and sent_ev.get("mean_ai_probability") is not None:
            st.caption(
                f"📝 **Sentence Evidence Context:** Analyzed {sent_ev.get('n_sentences_analyzed', 0)} sentences · "
                f"Mean sentence AI prob: **{sent_ev['mean_ai_probability']*100:.1f}%** · "
                f"AI-like sentence ratio (P≥0.50): **{sent_ev['ai_sentence_ratio']*100:.1f}%**"
            )

        st.divider()

        # --- Core 5-Feature Breakdown ---
        st.subheader("4. Core Five-Feature Forensic Breakdown")

        sigs = result.get("signals", {})
        feats = result.get("features", {})
        struct_d = result.get("structural_details", {})

        five_feature_rows = [
            {
                "Feature": "📈 Curvature (Fast-DetectGPT)",
                "Raw Value": f"{feats.get('curvature'):.4f}" if feats.get('curvature') is not None else "N/A",
                "Direction": "Higher → AI-like (+3.947 weight)",
                "Interpretation": "Negative log-probability curvature under distilgpt2",
            },
            {
                "Feature": "⚡ Burstiness",
                "Raw Value": f"{feats.get('burstiness'):.4f}" if feats.get('burstiness') is not None else "N/A (<5 sents)",
                "Direction": "Lower → AI-like (-1.128 weight)",
                "Interpretation": "Sentence length variation σ/μ (uniform length suggests AI)",
            },
            {
                "Feature": "🔤 Lexical Entropy",
                "Raw Value": f"{feats.get('lexical_entropy'):.4f}" if feats.get('lexical_entropy') is not None else "N/A",
                "Direction": "Lower → AI-like (-1.062 weight)",
                "Interpretation": "Shannon entropy of token distribution",
            },
            {
                "Feature": "🏗️ Structural Regularity",
                "Raw Value": f"{feats.get('structural_regularity'):.1f} / 100" if feats.get('structural_regularity') is not None else "N/A (<3 sents)",
                "Direction": "Uniformity (-0.037 weight)",
                "Interpretation": "Sentence-starter diversity & POS overlap composite",
            },
            {
                "Feature": "🚩 Cliché Density",
                "Raw Value": f"{feats.get('cliche_density'):.2f}%" if feats.get('cliche_density') is not None else "N/A",
                "Direction": "Higher → AI-like (+0.891 weight)",
                "Interpretation": "Frequency of 50+ overused AI idioms & buzzwords",
            },
        ]

        df_five = pd.DataFrame(five_feature_rows)
        st.dataframe(df_five, use_container_width=True)

        st.divider()

        # --- Explainability Tabs ---
        st.subheader("5. Explainability Details")
        exp_tab1, exp_tab2, exp_tab3 = st.tabs(["🚩 Detected AI Clichés", "📏 Sentence Rhythm", "🏗️ Structural Details"])

        with exp_tab1:
            detected_matches = []
            highlighted_text = raw_text
            for term in CLICHE_TERMS:
                pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
                matches = pattern.findall(raw_text)
                if matches:
                    detected_matches.extend(matches)
                    highlighted_text = pattern.sub(
                        lambda m: f'<mark class="cliche-highlight">{m.group(0)}</mark>',
                        highlighted_text,
                    )
            st.markdown(f'<div class="text-box">{highlighted_text}</div>', unsafe_allow_html=True)
            if detected_matches:
                st.error(f"Found **{len(detected_matches)} AI cliché match(es)**: `{', '.join(set(detected_matches))}`")
            else:
                st.success("✅ No known overused AI clichés detected in this text.")

        with exp_tab2:
            try:
                sentences = nltk.sent_tokenize(raw_text)
                sent_lengths = [len(s.split()) for s in sentences]
                if len(sentences) >= 5:
                    import numpy as np
                    mean_l = np.mean(sent_lengths)
                    std_l = np.std(sent_lengths)
                    burst = std_l / mean_l if mean_l > 0 else 0
                    st.write(f"• **Sentences detected**: {len(sentences)}")
                    st.write(f"• **Average sentence length**: {mean_l:.1f} words")
                    st.write(f"• **Sentence length variability (Burstiness $\\sigma/\\mu$)**: **{burst:.3f}**")
                    if burst < 0.35:
                        st.warning("⚠️ **Low Burstiness (<0.35)**: Sentence lengths are unnaturally uniform (typical of AI generation).")
                    else:
                        st.success("✅ **Normal/High Burstiness (≥0.35)**: Sentence lengths vary naturally (typical of human writing).")

                    df_sents = pd.DataFrame({
                        "Sentence #": [i + 1 for i in range(len(sentences))],
                        "Word Count": sent_lengths,
                        "Sentence Text": [s[:80] + ("..." if len(s) > 80 else "") for s in sentences],
                    })
                    st.dataframe(df_sents, use_container_width=True)
                else:
                    st.info(f"Only {len(sentences)} sentence(s) detected. Minimum 5 required for sentence burstiness calculation.")
            except Exception as e:
                st.write(f"Sentence analysis error: {e}")

        with exp_tab3:
            if struct_d:
                st.write(f"• **Sentence-starter diversity**: `{struct_d.get('starter_diversity', 'N/A')}`")
                st.write(f"• **POS pattern similarity**: `{struct_d.get('pos_similarity', 'N/A')}`")
                st.write(f"• **Discourse-marker density**: `{struct_d.get('discourse_density', 'N/A')}`")
            else:
                st.info("No structural detail available.")

        st.divider()

        # --- Advanced Expanders ---
        with st.expander("Advanced: T5 Robustness Self-Test"):
            stab_flag = result.get("stability_flag", "unknown")
            delta = result.get("paraphrase_delta", 0.0)
            if stab_flag == "stable":
                st.success(f"✅ **Stability: STABLE** — Paraphrase score delta: {delta:.2f} points (<= 15.0 threshold).")
            elif stab_flag == "unstable":
                st.warning(f"⚠️ **Stability: UNSTABLE** — Paraphrase caused a score shift of {delta:.2f} points (> 15.0 threshold).")
            elif stab_flag == "skipped":
                st.info("ℹ️ Robustness check was not enabled for this run.")
            else:
                st.info(f"ℹ️ Status: `{stab_flag}`.")

        with st.expander("Show Complete Raw Output (JSON)"):
            st.json(result)

        st.caption(f"⏱️ Total analysis time: **{elapsed:.2f} seconds**")
