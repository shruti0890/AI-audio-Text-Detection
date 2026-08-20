"""
app.py  (repository root)
=========================
Streamlit Application — AI Text & Audio Forensics Detection System

Presents an interface for:
  - Landing / Splash Hero Page (Premium Flowing Particle Wave Field Canvas)
  - Text Analysis  (5-feature text_forensics pipeline)
  - Audio Analysis (audio_forensics pipeline)

Run from the repository root:
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
import sys
import time
import uuid
from pathlib import Path

import nltk
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── Path setup ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion_integration import run_full_pipeline

# ── Session-state initialisation ─────────────────────────────────────────────
for _key, _default in (
    ("app_page", "landing"),       # "landing" | "analysis"
    ("analysis_mode", "Text"),     # "Text" | "Audio"
    ("audio_result", None),
    ("audio_file_id", None),
    ("cross_modal_text_result", None),
    ("cross_modal_audio_score", None),
    ("cross_modal_audio_result", None),
    ("cross_modal_reasoner_result", None),
):
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── Optional: sentence-level highlighting (text forensics) ───────────────────
try:
    from text_forensics.signals.sentence_scorer import score_sentences
    from text_forensics.pipeline import _load_baseline_stats
    from text_forensics.signals.cliche_scanner import CLICHE_TERMS
    _TEXT_EXTRAS_AVAILABLE = True
except ImportError:
    _TEXT_EXTRAS_AVAILABLE = False

# ── NLTK punkt check ──────────────────────────────────────────────────────────
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forensic AI — Text & Audio Forensics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed" if st.session_state["app_page"] == "landing" else "expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: verdict HTML card
# ─────────────────────────────────────────────────────────────────────────────
def _verdict_card(verdict: str, score: float, subtitle: str = "", model_name: str = "Five-Feature Logistic Regression") -> str:
    v = verdict.lower()
    if v == "ai" or (("likely ai" not in v) and ("ai" in v or "deepfake" in v or "generated" in v)):
        css, title_css = "verdict-ai", "verdict-title-ai"
        icon = "🤖"
    elif "likely ai" in v:
        css, title_css = "verdict-likely-ai", "verdict-title-likely-ai"
        icon = "🤖"
    elif "likely human" in v:
        css, title_css = "verdict-likely-human", "verdict-title-likely-human"
        icon = "👤"
    elif "human" in v or "real" in v:
        css, title_css = "verdict-human", "verdict-title-human"
        icon = "👤"
    else:
        css, title_css = "verdict-uncertain", "verdict-title-uncertain"
        icon = "🔶"
    sub = f'<div class="verdict-subtitle" style="color:#475569;">{subtitle}</div>' if subtitle else ""
    return (
        f'<div class="{css}">'
        f'<div class="{title_css}">{icon} {verdict}</div>'
        f'<div class="verdict-subtitle">AI Probability: <b>{score:.1f}%</b></div>'
        f'<div style="font-size:0.85rem;color:#64748B;margin-top:0.25rem;">Model: <b>{model_name}</b></div>'
        f'{sub}'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: render sentence-level highlighted text
# ─────────────────────────────────────────────────────────────────────────────
def _render_sentence_highlights(text: str, baseline: dict | None) -> None:
    if not _TEXT_EXTRAS_AVAILABLE:
        st.text(text)
        return
    sentence_analysis = score_sentences(text, baseline)
    html_spans = []
    for s_item in sentence_analysis:
        s_text = s_item["sentence"]
        c_score = s_item.get("curvature_score")
        if c_score is None:
            span = f'<span class="sent-short" title="Insufficient context">{s_text}</span>'
        elif c_score >= 50:
            span = f'<span class="sent-high-ai" title="AI score: {c_score:.1f}/100">{s_text}</span>'
        else:
            span = f'<span class="sent-low-ai" title="AI score: {c_score:.1f}/100">{s_text}</span>'
        html_spans.append(span)
    st.markdown(f'<div class="text-box">{" ".join(html_spans)}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: render cliché highlights
# ─────────────────────────────────────────────────────────────────────────────
def _render_cliche_highlights(text: str) -> None:
    if not _TEXT_EXTRAS_AVAILABLE:
        st.text(text)
        return
    highlighted = text
    detected = []
    for term in CLICHE_TERMS:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        if pattern.search(text):
            detected.append(term)
            highlighted = pattern.sub(
                lambda m: f'<mark class="cliche-highlight">{m.group(0)}</mark>',
                highlighted,
            )
    st.markdown(f'<div class="text-box">{highlighted}</div>', unsafe_allow_html=True)
    if detected:
        st.error(f"Found **{len(detected)} AI cliché(s)**: `{', '.join(set(detected))}`")
    else:
        st.success("✅ No AI clichés detected.")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: render text results panel
# ─────────────────────────────────────────────────────────────────────────────
def _render_text_panel(result: dict, raw_text: str) -> None:
    text_score = result["text_score"]
    text_verdict = result["text_verdict"]
    model_name = result.get("text_model_used", "Five-Feature Logistic Regression")
    model_display = "Five-Feature Logistic Regression" if "five_feature" in model_name else "Corrected Four-Feature Baseline"

    st.markdown('<div class="section-header">📝 Text Forensics Results</div>', unsafe_allow_html=True)
    st.markdown(_verdict_card(text_verdict, text_score, model_name=model_display), unsafe_allow_html=True)

    # ── Stage 2: Experimental AI-Generator Attribution ──
    if text_verdict in ["Likely AI", "AI"]:
        try:
            from text_forensics.calibration.generator_attribution_experiment.attribution_pipeline import analyze_text_with_attribution
            attr_res = analyze_text_with_attribution(raw_text, run_robustness=False)
            attr_dist = attr_res.get("generator_attribution")
            pred_gen = attr_res.get("predicted_generator", "Unknown / Other AI")
            gen_conf = attr_res.get("generator_confidence", 0.0)
            gen_verdict = attr_res.get("generator_verdict", "")
            disclaimer = attr_res.get("attribution_disclaimer", "")

            if attr_dist:
                st.markdown("##### 🎯 AI-Generator Attribution (Stage 2 Experimental)")
                col_g1, col_g2 = st.columns([1, 1])
                with col_g1:
                    st.metric("Estimated Model Family", pred_gen, delta=f"Confidence: {gen_conf*100:.1f}%", delta_color="normal")
                with col_g2:
                    st.caption(f"Status: **{gen_verdict}**")
                    st.caption(f"⚖️ *{disclaimer}*")

                g_col1, g_col2, g_col3, g_col4 = st.columns(4)
                with g_col1:
                    st.write(f"**ChatGPT / OpenAI**: `{attr_dist.get('chatgpt', 0.0)*100:.1f}%`")
                    st.progress(float(attr_dist.get('chatgpt', 0.0)))
                with g_col2:
                    st.write(f"**Google Gemini**: `{attr_dist.get('gemini', 0.0)*100:.1f}%`")
                    st.progress(float(attr_dist.get('gemini', 0.0)))
                with g_col3:
                    st.write(f"**Anthropic Claude**: `{attr_dist.get('claude', 0.0)*100:.1f}%`")
                    st.progress(float(attr_dist.get('claude', 0.0)))
                with g_col4:
                    st.write(f"**Other / Unknown AI**: `{attr_dist.get('other_ai', 0.0)*100:.1f}%`")
                    st.progress(float(attr_dist.get('other_ai', 0.0)))
        except Exception:
            pass

    if result.get("text_signal_agreement") == "disagreement":
        st.error(
            "⚠️ **Signals Disagree Significantly**: Individual detectors are giving "
            "conflicting evidence (sub-score spread > 40 points). Review the feature breakdown below."
        )

    st.divider()
    st.markdown("**Sentence-Level Evidence** *(red = AI-likely, green = human-likely)*")
    baseline = _load_baseline_stats() if _TEXT_EXTRAS_AVAILABLE else None
    _render_sentence_highlights(raw_text, baseline)
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Core Five Features", "🚩 Cliché Evidence", "📏 Rhythm Analysis"])

    with tab1:
        feats = result.get("text_features", {}) or {}
        feature_rows = [
            {
                "Feature": "📈 Curvature (Fast-DetectGPT)",
                "Raw Value": f"{feats.get('curvature'):.4f}" if feats.get('curvature') is not None else "N/A",
                "Direction": "Higher → AI-like (+3.947 weight)",
                "Interpretation": "Negative log-probability discrepancy under distilgpt2",
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
        st.dataframe(pd.DataFrame(feature_rows), use_container_width=True)
        st.caption("ℹ️ *Raw forensic feature values fed into the Logistic Regression model, not individual AI probabilities.*")

    with tab2:
        _render_cliche_highlights(raw_text)

    with tab3:
        try:
            sentences = nltk.sent_tokenize(raw_text)
            sent_lengths = [len(s.split()) for s in sentences]
            if len(sentences) >= 5:
                import numpy as np
                mean_l = np.mean(sent_lengths)
                std_l = np.std(sent_lengths)
                burst = std_l / mean_l if mean_l > 0 else 0
                st.write(f"• **Sentences**: {len(sentences)}")
                st.write(f"• **Avg sentence length**: {mean_l:.1f} words")
                st.write(f"• **Burstiness (σ/μ)**: `{burst:.3f}`")
                if burst < 0.35:
                    st.warning("⚠️ **Low Burstiness (<0.35)**: Unnaturally uniform sentence lengths.")
                else:
                    st.success("✅ **Normal Burstiness (≥0.35)**: Natural length variation.")
            else:
                st.info(f"Only {len(sentences)} sentence(s) — minimum 5 needed for burstiness.")
        except Exception as e:
            st.write(f"Rhythm analysis error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: render audio results panel
# ─────────────────────────────────────────────────────────────────────────────
def _render_audio_panel(result: dict) -> None:
    audio_score = result["audio_score"]
    audio_verdict = result["audio_verdict"]
    logit_fake = result["logit_fake"]
    logit_real = result["logit_real"]
    logit_diff = logit_fake - logit_real
    transcript = result["transcript"]
    threshold = result.get("audio_decision_threshold_pct", 50.0)
    temperature = result.get("audio_temperature", 1.15)
    n_windows = result.get("audio_n_windows", 1)
    window_scores = result.get("audio_window_scores", [])
    conf_tiers = result.get("audio_confidence_tiers", {})

    st.markdown('<div class="section-header section-header-audio">🎙️ Stage 1 — Audio Detection</div>', unsafe_allow_html=True)
    st.markdown(_verdict_card(audio_verdict, audio_score, model_name="wav2vec2-deepfake-voice-detector"), unsafe_allow_html=True)
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Deepfake Score (Max)", f"{audio_score:.2f}%")
    col2.metric("Logit (Fake)", f"{logit_fake:.4f}")
    col3.metric("Logit (Real)", f"{logit_real:.4f}")
    col4.metric("Δ (Fake−Real)", f"{logit_diff:+.4f}")

    st.markdown("**Synthetic Risk Bar:**")
    st.progress(min(max(audio_score / 100.0, 0.0), 1.0))

    mean_sc = result.get("audio_mean_score", audio_score)
    med_sc = result.get("audio_median_score", audio_score)
    dur_sec = result.get("audio_duration_seconds")
    batch_sz = result.get("audio_inference_batch_size", 1)
    win_sz = result.get("audio_window_size_seconds", 5)
    win_ov = result.get("audio_window_overlap_seconds", 1)
    win_step = result.get("audio_window_step_seconds", 4)
    audio_meta = result.get("audio_metadata", {}) or {}

    st.markdown(
        f"> **Scoring Formula (T={temperature}):** "
        f"$$S_{{Audio}} = \\text{{Sigmoid}}\\!\\left(\\frac{{{logit_fake:.4f}-({logit_real:.4f})}}{{{temperature}}}\\right)"
        f"\\times 100 = {audio_score:.2f}\\%$$\n"
        f"> **Decision Threshold:** `{threshold:.1f}%` | **Aggregation:** `max_risk` | **Windows:** `{n_windows}`"
    )

    with st.expander("📊 Audio Specs & Sliding-Window Details", expanded=True):
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("File Format", f"{audio_meta.get('original_format', 'AUDIO')}")
        mcol2.metric("Duration", f"{audio_meta.get('original_duration_seconds', dur_sec or 0.0):.1f} s")
        mcol3.metric("Windows Analyzed", f"{n_windows}")
        mcol4.metric("Inference Batch Size", f"{batch_sz} (bounded)")
        sc_col1, sc_col2, sc_col3 = st.columns(3)
        sc_col1.metric("Max Window Score", f"{audio_score:.2f}%")
        sc_col2.metric("Mean Window Score", f"{mean_sc:.2f}%")
        sc_col3.metric("Median Window Score", f"{med_sc:.2f}%")
        st.caption(
            f"⚙️ Window Size = `{win_sz}s` | Overlap = `{win_ov}s` | Step = `{win_step}s` | "
            f"SR = `{audio_meta.get('normalized_sample_rate', 16000)} Hz mono`"
        )
        if n_windows > 1:
            st.caption(f"🔍 Per-window scores: {[f'{s:.1f}%' for s in window_scores]}")

    st.divider()

    vad_time = result.get("audio_vad_time")
    asr_time = result.get("audio_asr_time")
    deepfake_time = result.get("audio_deepfake_time")
    total_time = result.get("audio_total_time")

    if any(v is not None for v in [vad_time, asr_time, deepfake_time]):
        st.markdown("**Process Timing Breakdown**")
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        if vad_time is not None: t_col1.metric("Silero VAD", f"{vad_time:.2f} s")
        if asr_time is not None: t_col2.metric("Whisper ASR", f"{asr_time:.2f} s")
        if deepfake_time is not None: t_col3.metric("Deepfake Score", f"{deepfake_time:.2f} s")
        if total_time is not None: t_col4.metric("Total Pipeline", f"{total_time:.2f} s")
        st.divider()

    st.markdown("**Speech-to-Text Transcript (Whisper-Tiny ASR)**")
    has_transcript = bool(transcript and transcript.strip())
    transcript_word_count = len(transcript.split()) if has_transcript else 0
    transcript_long_enough = transcript_word_count > 30

    if has_transcript:
        st.text_area("Transcribed Audio Content", value=transcript, height=110, disabled=True)
        st.caption(f"📊 {transcript_word_count} words | {len(transcript)} characters")
    else:
        st.warning("⚠️ No spoken text transcribed (silence, noise, or non-speech audio).")

    st.markdown("---")
    st.markdown('<div class="section-header section-header-audio">🧬 Stage 2 — Cross-Modality Analysis</div>', unsafe_allow_html=True)

    if not has_transcript:
        st.info("ℹ️ A valid speech transcript is required to perform Cross-Modality Analysis.")
    elif not transcript_long_enough:
        st.error(
            f"❌ **Transcript too short** — `{transcript_word_count}` words transcribed. "
            "The Five-Feature Text pipeline requires **more than 30 words** for reliable results."
        )
    else:
        transcribed_text_analyzed = (
            st.session_state.get("cross_modal_text_result") is not None
            and st.session_state.get("cross_modal_audio_score") == audio_score
        )
        col_btn, col_hint = st.columns([1, 2])
        with col_btn:
            analyse_text_btn = st.button(
                "🔬 Analyse Transcribed Text",
                type="secondary" if transcribed_text_analyzed else "primary",
                key="btn_analyse_transcribed_text",
                disabled=not has_transcript,
                use_container_width=True,
            )
        with col_hint:
            if not transcribed_text_analyzed:
                st.caption("Click to run the **Five-Feature Text Forensics Pipeline** on the transcript and compare modalities.")
            else:
                st.caption("✅ Transcript analysis and cross-modal comparison complete.")

        if analyse_text_btn:
            with st.spinner("Analysing transcribed text with Five-Feature Text Forensics Pipeline..."):
                try:
                    from text_forensics.pipeline import analyze_text
                    from cross_modal_reasoner import evaluate_cross_modality
                    text_res = analyze_text(transcript.strip(), run_robustness=False)
                    st.session_state["cross_modal_text_result"] = text_res
                    st.session_state["cross_modal_audio_score"] = audio_score
                    st.session_state["cross_modal_audio_result"] = result
                    st.session_state["cross_modal_reasoner_result"] = evaluate_cross_modality(result, text_res)
                    st.rerun()
                except Exception as ex:
                    st.error(f"❌ Text analysis failed: `{ex}`")

        cm_reasoner = st.session_state.get("cross_modal_reasoner_result")
        cm_text_res = st.session_state.get("cross_modal_text_result")

        if cm_reasoner and cm_text_res and st.session_state.get("cross_modal_audio_score") == audio_score:
            st.markdown("#### 📝 Transcribed Text Forensic Evaluation")
            t_c1, t_c2, t_c3 = st.columns(3)
            t_c1.metric("Text AI Score", f"{cm_reasoner['text']['score']:.2f}%")
            t_c2.metric("Decision Threshold", f"{cm_reasoner['text']['threshold']:.1f}%")
            t_c3.metric("Text Classification", f"{cm_reasoner['text']['classification']} ({cm_reasoner['text']['verdict']})")

            box_class = "cross-modal-consistent" if cm_reasoner["consistent"] else "cross-modal-conflict"
            title_class = "cross-modal-title-consistent" if cm_reasoner["consistent"] else "cross-modal-title-conflict"
            st.markdown(
                f'<div class="{box_class}">'
                f'<div class="{title_class}">{cm_reasoner["title"]}</div>'
                f'<div class="cross-modal-subtitle">{cm_reasoner["classification"]}</div>'
                f'<div class="cross-modal-desc">{cm_reasoner["description"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### 🔍 Cross-Modality Evidence Comparison")
            ev_col1, ev_col2 = st.columns(2)
            with ev_col1:
                st.markdown("**🎙️ Audio Modality**")
                st.write(f"• **Score:** `{cm_reasoner['audio']['score']:.2f}%`")
                st.write(f"• **Threshold:** `{cm_reasoner['audio']['threshold']:.1f}%`")
                st.write(f"• **Classification:** `{cm_reasoner['audio']['classification']}` ({cm_reasoner['audio']['verdict']})")
                st.write("• **Model:** `wav2vec2-deepfake-voice-detector`")
            with ev_col2:
                st.markdown("**📝 Text Modality (Transcript)**")
                st.write(f"• **Score:** `{cm_reasoner['text']['score']:.2f}%`")
                st.write(f"• **Threshold:** `{cm_reasoner['text']['threshold']:.1f}%`")
                st.write(f"• **Classification:** `{cm_reasoner['text']['classification']}` ({cm_reasoner['text']['verdict']})")
                st.write("• **Model:** `Five-Feature Logistic Regression`")
        elif not transcribed_text_analyzed:
            st.info("💡 Click **Analyse Transcribed Text** above to evaluate the transcript and compare cross-modal evidence.")

    st.divider()

    abs_diff = abs(logit_diff)
    extreme_tier  = conf_tiers.get("extreme_logit_margin", 3.0)
    high_tier     = conf_tiers.get("high_logit_margin", 1.5)
    moderate_tier = conf_tiers.get("moderate_logit_margin", 0.5)
    confidence = (
        "Extreme Confidence" if abs_diff > extreme_tier else
        "High Confidence" if abs_diff > high_tier else
        "Moderate Confidence" if abs_diff > moderate_tier else
        "Borderline / Low Confidence"
    )

    with st.expander("📋 Audio Classification Explanation"):
        if audio_score >= 75.0:
            tier_label, tier_range = "Authentic AI Voice", "75–100%"
        elif audio_score >= 56.0:
            tier_label, tier_range = "Likely AI Voice", "56–74%"
        elif audio_score >= 36.0:
            tier_label, tier_range = "Likely Human Voice", "36–55%"
        else:
            tier_label, tier_range = "Authentic Human Voice", "0–35%"

        if audio_score >= 56.0:
            st.markdown(
                f"**Why classified as {tier_label}?**\n\n"
                f"- **Tier:** Score `{audio_score:.2f}%` falls in **{tier_label}** range ({tier_range}).\n"
                f"- **Logit Dominance**: `logit_fake` ({logit_fake:.4f}) > `logit_real` ({logit_real:.4f}) by **{logit_diff:+.4f}**.\n"
                f"- **Sigmoid Mapping**: Deepfake risk score **{audio_score:.2f}%** (≥{threshold:.1f}% threshold).\n"
                f"- **Confidence**: **{confidence}** (|Δ| = `{abs_diff:.4f}`).\n"
                f"- wav2vec2 detected spectral artifacts typical of synthetic speech."
            )
        else:
            st.markdown(
                f"**Why classified as {tier_label}?**\n\n"
                f"- **Tier:** Score `{audio_score:.2f}%` falls in **{tier_label}** range ({tier_range}).\n"
                f"- **Logit Dominance**: `logit_real` ({logit_real:.4f}) ≥ `logit_fake` ({logit_fake:.4f}); margin **{logit_diff:+.4f}**.\n"
                f"- **Sigmoid Mapping**: Deepfake risk score **{audio_score:.2f}%** (<{threshold:.1f}% threshold).\n"
                f"- **Confidence**: **{confidence}** (|Δ| = `{abs_diff:.4f}`).\n"
                f"- Acoustic features align with natural human vocal tract resonances."
            )


# ═════════════════════════════════════════════════════════════════════════════
# VIEW ROUTING: LANDING vs ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state["app_page"] == "landing":
    # ── Global Full-Bleed Dark Viewport CSS for Streamlit Root ────────────────
    st.markdown("""
    <style>
    /* Dark Theme across all Streamlit container levels */
    html, body, [data-testid="stAppViewContainer"], .stApp, .main, .block-container, [data-testid="stHeader"] {
        background-color: #040811 !important;
        background: #040811 !important;
        color: #E2E8F0 !important;
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100vw !important;
        width: 100% !important;
        overflow-x: hidden !important;
    }
    #MainMenu, footer, header, [data-testid="stSidebar"] {
        display: none !important;
        visibility: hidden !important;
    }
    iframe {
        display: block !important;
        width: 100% !important;
        height: 100vh !important;
        min-height: 720px !important;
        border: none !important;
    }
    /* Hidden Streamlit button triggered by iframe CTA click */
    div[data-testid="stButton"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Premium Flowing Particle Wave Canvas HTML ─────────────────────────────
    LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html, body {
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: #030611;
    color: #E2E8F0;
    font-family: 'Space Grotesk', 'Inter', sans-serif;
  }

  /* ── Fullscreen Hero Viewport ── */
  .hero-viewport {
    position: relative;
    width: 100vw;
    height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 2.2rem 4.5rem 2.5rem 4.5rem;
    overflow: hidden;
    background: radial-gradient(circle at 75% 45%, #081226 0%, #050B18 50%, #030611 90%);
  }

  /* ── Flowing Particle Wave Canvas ── */
  #wave-canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 1;
    pointer-events: none;
  }

  /* ── Ambient Radial Glows ── */
  .ambient-glow-cyan {
    position: absolute;
    top: 15%;
    right: 15%;
    width: 650px;
    height: 650px;
    background: radial-gradient(circle, rgba(0, 240, 255, 0.08) 0%, rgba(59, 130, 246, 0.03) 50%, transparent 75%);
    filter: blur(80px);
    pointer-events: none;
    z-index: 1;
  }

  .ambient-glow-purple {
    position: absolute;
    bottom: 8%;
    right: 22%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(147, 51, 234, 0.09) 0%, rgba(236, 72, 153, 0.03) 50%, transparent 75%);
    filter: blur(90px);
    pointer-events: none;
    z-index: 1;
  }

  /* ── Minimal Top Navigation ── */
  .landing-nav {
    position: relative;
    z-index: 10;
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
  }

  .landing-logo {
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }

  .landing-nav-links {
    display: flex;
    gap: 2.8rem;
    list-style: none;
  }

  .landing-nav-links span {
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #94A3B8;
    cursor: default;
    transition: color 0.2s ease;
  }

  .landing-nav-links span:hover {
    color: #38BDF8;
  }

  /* ── Scattered Background Forensic Words ── */
  .ambient-words {
    position: absolute;
    inset: 0;
    pointer-events: none;
    user-select: none;
    z-index: 2;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
  }

  .ambient-words span {
    position: absolute;
    color: #94A3B8;
    transition: opacity 0.5s ease;
  }

  /* ── Hero Left Body ── */
  .hero-body {
    position: relative;
    z-index: 10;
    max-width: 650px;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
  }

  .hero-eyebrow {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1.4rem;
  }

  .hero-eyebrow-line {
    width: 36px;
    height: 2px;
    background: linear-gradient(90deg, #00F0FF, #818CF8);
    box-shadow: 0 0 10px rgba(0, 240, 255, 0.6);
  }

  .hero-eyebrow-text {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #38BDF8;
  }

  .hero-title-main {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(3.6rem, 6.2vw, 6.2rem);
    font-weight: 800;
    line-height: 0.96;
    letter-spacing: -0.03em;
    margin: 0 0 1.3rem 0;
  }

  .hero-title-white {
    display: block;
    color: #FFFFFF;
    text-shadow: 0 0 30px rgba(255, 255, 255, 0.2);
  }

  .hero-title-gradient {
    display: block;
    background: linear-gradient(105deg, #00F0FF 0%, #38BDF8 40%, #818CF8 75%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 25px rgba(0, 240, 255, 0.4));
  }

  .hero-subhead {
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94A3B8;
    margin: 0 0 1.1rem 0;
  }

  .hero-desc-text {
    font-size: 1.18rem;
    font-weight: 400;
    line-height: 1.7;
    color: #CBD5E1;
    max-width: 500px;
    margin: 0 0 2.2rem 0;
  }

  /* ── Sleek Glowing CTA Button ── */
  .cta-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.95rem 2.8rem;
    background: linear-gradient(115deg, #0284C7 0%, #2563EB 50%, #7C3AED 100%);
    color: #FFFFFF;
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 9999px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.02rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    box-shadow: 0 0 30px rgba(56, 189, 248, 0.45), 0 8px 25px rgba(0, 0, 0, 0.6);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    outline: none;
    text-decoration: none;
  }

  .cta-btn:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 0 45px rgba(124, 58, 237, 0.7), 0 12px 30px rgba(0, 0, 0, 0.7);
    border-color: rgba(255, 255, 255, 0.6);
  }

  /* ── Bottom Information Metrics Strip ── */
  .landing-footer-strip {
    position: relative;
    z-index: 10;
    display: flex;
    gap: 3.5rem;
    align-items: center;
    border-top: 1px solid rgba(148, 163, 184, 0.12);
    padding-top: 1.3rem;
  }

  .footer-stat-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 0.25rem;
  }

  .footer-stat-value {
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    color: #94A3B8;
    text-transform: uppercase;
  }

  .footer-stat-divider {
    width: 1px;
    height: 30px;
    background: rgba(148, 163, 184, 0.18);
  }
</style>
</head>
<body>

<div class="hero-viewport">
  <!-- HTML5 Canvas for Continuous Flowing Particle Wave Field -->
  <canvas id="wave-canvas"></canvas>

  <!-- Ambient Glow Orbs -->
  <div class="ambient-glow-cyan"></div>
  <div class="ambient-glow-purple"></div>

  <!-- Ambient Forensic Background Text -->
  <div class="ambient-words">
    <span style="top:12%;left:4%;font-size:0.82rem;opacity:0.09;transform:rotate(-6deg);">ENTROPY</span>
    <span style="top:25%;left:8%;font-size:0.72rem;opacity:0.07;transform:rotate(4deg);">SIGNAL</span>
    <span style="top:42%;left:3%;font-size:0.92rem;opacity:0.08;transform:rotate(-4deg);color:#38BDF8;">SYNTHETIC</span>
    <span style="top:62%;left:6%;font-size:0.72rem;opacity:0.06;">LINGUISTIC</span>
    <span style="top:80%;left:4%;font-size:0.82rem;opacity:0.08;transform:rotate(5deg);color:#818CF8;">AUTHENTIC</span>
    <span style="top:92%;left:9%;font-size:0.72rem;opacity:0.07;">PATTERN</span>
    <span style="top:10%;right:6%;font-size:0.88rem;opacity:0.08;transform:rotate(6deg);color:#00F0FF;">CURVATURE</span>
    <span style="top:28%;right:12%;font-size:0.72rem;opacity:0.07;">DETECTION</span>
    <span style="top:74%;right:7%;font-size:0.82rem;opacity:0.08;transform:rotate(-5deg);">ANALYSIS</span>
    <span style="top:88%;right:14%;font-size:0.72rem;opacity:0.07;color:#C084FC;">SPEECH</span>
    <span style="top:18%;left:46%;font-size:0.68rem;opacity:0.05;">MODEL</span>
    <span style="top:68%;left:44%;font-size:0.72rem;opacity:0.06;transform:rotate(-3deg);">HUMAN</span>
  </div>

  <!-- Minimal Top Navigation -->
  <div class="landing-nav">
    <div class="landing-logo">
      <span>🔬</span>
      <span>FORENSIC AI</span>
    </div>
    <div class="landing-nav-links">
      <span>TEXT</span>
      <span>AUDIO</span>
      <span>ABOUT</span>
    </div>
  </div>

  <!-- Hero Content Block -->
  <div class="hero-body">
    <div class="hero-eyebrow">
      <div class="hero-eyebrow-line"></div>
      <span class="hero-eyebrow-text">AI DETECTION SYSTEM</span>
    </div>

    <div class="hero-title-main">
      <span class="hero-title-white">FORENSIC</span>
      <span class="hero-title-gradient">AI</span>
    </div>

    <p class="hero-subhead">AI TEXT &amp; AUDIO FORENSICS</p>

    <p class="hero-desc-text">
      Analyze linguistic and acoustic patterns to uncover AI-generated content.
    </p>

    <button class="cta-btn" id="landing-cta-btn" onclick="startApp()">
      GET STARTED &rarr;
    </button>
  </div>

  <!-- Bottom Information Strip -->
  <div class="landing-footer-strip">
    <div>
      <div class="footer-stat-label">Text Forensics</div>
      <div class="footer-stat-value">5 Core Linguistic Signals</div>
    </div>
    <div class="footer-stat-divider"></div>
    <div>
      <div class="footer-stat-label">Audio Forensics</div>
      <div class="footer-stat-value">VAD • ASR • Wav2Vec2</div>
    </div>
    <div class="footer-stat-divider"></div>
    <div>
      <div class="footer-stat-label">Accuracy Engine</div>
      <div class="footer-stat-value">Calibrated • Reliable • Explainable</div>
    </div>
  </div>
</div>

<script>
  // ── Navigation Trigger ───────────────────────────────────────────────────────
  function startApp() {
    try {
      const parentDoc = window.parent.document;
      const btn = parentDoc.querySelector('[data-testid="stButton"] button');
      if (btn) {
        btn.click();
      }
    } catch (e) {
      console.error(e);
    }
  }

  // ── Flowing Digital Particle Wave Field Canvas Engine ────────────────────────
  const canvas = document.getElementById('wave-canvas');
  const ctx = canvas.getContext('2d');

  let width, height;
  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  // Color Palette definitions
  const colors = [
    { r: 0,   g: 240, b: 255 }, // Cyan #00F0FF
    { r: 56,  g: 189, b: 248 }, // Sky Blue #38BDF8
    { r: 59,  g: 130, b: 246 }, // Electric Blue #3B82F6
    { r: 99,  g: 102, b: 241 }, // Indigo #6366F1
    { r: 139, g: 92,  b: 246 }, // Violet #8B5CF6
    { r: 192, g: 132, b: 252 }, // Purple #C084FC
    { r: 236, g: 72,  b: 153 }  // Magenta #EC4899
  ];

  // Ribbon harmonic wave parameters (smooth fluid mathematical field)
  const ribbons = [
    { yOffset: 0.46, amp: 75, freq: 0.0018, speed: 0.012, colorIdx: 0, spread: 28 },
    { yOffset: 0.50, amp: 90, freq: 0.0015, speed: 0.009, colorIdx: 1, spread: 35 },
    { yOffset: 0.54, amp: 80, freq: 0.0020, speed: 0.014, colorIdx: 2, spread: 30 },
    { yOffset: 0.58, amp: 105, freq: 0.0014, speed: 0.008, colorIdx: 4, spread: 45 },
    { yOffset: 0.62, amp: 85, freq: 0.0022, speed: 0.011, colorIdx: 5, spread: 32 },
    { yOffset: 0.66, amp: 70, freq: 0.0017, speed: 0.015, colorIdx: 6, spread: 25 }
  ];

  // Generate 450 flowing data particles
  const particleCount = 450;
  const particles = [];

  for (let i = 0; i < particleCount; i++) {
    particles.push({
      x: Math.random() * window.innerWidth,
      ribbonIdx: Math.floor(Math.random() * ribbons.length),
      offsetY: (Math.random() - 0.5) * 1.6, // Vertical spread within ribbon
      size: Math.random() * 2.2 + 0.8,
      speedX: Math.random() * 1.4 + 0.6,
      opacity: Math.random() * 0.7 + 0.3,
      pulseSpeed: Math.random() * 0.03 + 0.01,
      pulsePhase: Math.random() * Math.PI * 2
    });
  }

  let time = 0;

  function render() {
    time += 1;
    ctx.clearRect(0, 0, width, height);

    // Wave computation function
    function getWaveY(x, ribbon) {
      const baseY = height * ribbon.yOffset;
      // Multi-frequency harmonic superposition for organic fluid motion
      const w1 = Math.sin(x * ribbon.freq + time * ribbon.speed) * ribbon.amp;
      const w2 = Math.cos(x * ribbon.freq * 0.5 - time * ribbon.speed * 0.7) * (ribbon.amp * 0.4);
      const w3 = Math.sin(x * 0.0008 + time * 0.005) * 25;
      return baseY + w1 + w2 + w3;
    }

    // 1. Draw glowing flowing ribbon paths
    ribbons.forEach((ribbon, rIdx) => {
      ctx.beginPath();
      const col = colors[ribbon.colorIdx];
      const startX = width * 0.25; // Wave field starts around left 25% and expands across right
      
      for (let x = startX; x <= width + 20; x += 15) {
        const y = getWaveY(x, ribbon);
        if (x === startX) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.strokeStyle = `rgba(${col.r}, ${col.g}, ${col.b}, 0.18)`;
      ctx.lineWidth = 1.6;
      ctx.stroke();

      // Subtle second harmonic filament
      ctx.beginPath();
      for (let x = startX + 50; x <= width + 20; x += 20) {
        const y = getWaveY(x, ribbon) + Math.sin(x * 0.01 + time * 0.02) * 6;
        if (x === startX + 50) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.strokeStyle = `rgba(${col.r}, ${col.g}, ${col.b}, 0.09)`;
      ctx.lineWidth = 0.8;
      ctx.stroke();
    });

    // 2. Render and animate flowing data particles
    particles.forEach(p => {
      // Advance particle position
      p.x += p.speedX;
      if (p.x > width + 20) {
        p.x = width * 0.22 + Math.random() * 50; // Recycle to wave origin
        p.ribbonIdx = Math.floor(Math.random() * ribbons.length);
      }

      const ribbon = ribbons[p.ribbonIdx];
      const col = colors[ribbon.colorIdx];
      const waveCenterY = getWaveY(p.x, ribbon);
      const y = waveCenterY + p.offsetY * ribbon.spread;

      // Distance fade: transparent on far left, bright across right
      const xProgress = Math.max(0, Math.min(1, (p.x - width * 0.2) / (width * 0.35)));
      p.pulsePhase += p.pulseSpeed;
      const pulsingAlpha = (Math.sin(p.pulsePhase) * 0.25 + 0.75) * p.opacity * xProgress;

      if (pulsingAlpha > 0.01) {
        // Glowing core
        ctx.beginPath();
        ctx.arc(p.x, y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${col.r}, ${col.g}, ${col.b}, ${pulsingAlpha})`;
        ctx.shadowColor = `rgba(${col.r}, ${col.g}, ${col.b}, 0.8)`;
        ctx.shadowBlur = 8;
        ctx.fill();

        // Particle subtle motion trail
        ctx.beginPath();
        ctx.moveTo(p.x, y);
        ctx.lineTo(p.x - p.speedX * 4, y - (p.offsetY * 1.5));
        ctx.strokeStyle = `rgba(${col.r}, ${col.g}, ${col.b}, ${pulsingAlpha * 0.3})`;
        ctx.lineWidth = p.size * 0.6;
        ctx.shadowBlur = 0;
        ctx.stroke();
      }
    });

    ctx.shadowBlur = 0; // Reset shadow for next frame
    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
</script>
</body>
</html>"""

    # Render inside components.html with full height
    components.html(LANDING_HTML, height=920, scrolling=False)

    # Hidden Streamlit button that handles the transition to the analysis page
    if st.button("PROCEED_TO_ANALYSIS", key="btn_landing_hidden_trigger"):
        st.session_state["app_page"] = "analysis"
        st.rerun()

else:
    # ═════════════════════════════════════════════════════════════════════════
    # ── ANALYSIS PAGE ─────────────────────────────────────────────────────────
    # ═════════════════════════════════════════════════════════════════════════

    # Analysis Page Styling
    analysis_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem !important; }

.analysis-header-title { font-size: 2.0rem; font-weight: 800; color: #0F172A; margin: 0 0 0.15rem 0; }
.analysis-header-subtitle { font-size: 1.0rem; color: #64748B; margin: 0 0 1.2rem 0; }

.mode-card {
    background: #FFFFFF; border: 2px solid #E2E8F0; border-radius: 16px;
    padding: 1.3rem 1.5rem 1rem 1.5rem; margin-bottom: 0.6rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.mode-card-active {
    border-color: #3B82F6; background: linear-gradient(160deg, #F8FAFC 0%, #EFF6FF 100%);
    box-shadow: 0 6px 20px rgba(59,130,246,0.14);
}
.mode-card-active-audio {
    border-color: #8B5CF6; background: linear-gradient(160deg, #F8FAFC 0%, #FAF5FF 100%);
    box-shadow: 0 6px 20px rgba(139,92,246,0.14);
}
.mode-card-title { font-size: 1.2rem; font-weight: 800; color: #0F172A; margin-bottom: 0.3rem; }
.mode-card-desc { font-size: 0.93rem; color: #475569; line-height: 1.5; margin-bottom: 0.8rem; }

.verdict-ai { background: linear-gradient(135deg,#FEF2F2 0%,#FFE4E6 100%); border: 2px solid #EF4444; border-radius:14px; padding:1.4rem 1.6rem; text-align:center; margin-bottom:1rem; }
.verdict-likely-ai { background: linear-gradient(135deg,#FFF7ED 0%,#FFEDD5 100%); border: 2px solid #F97316; border-radius:14px; padding:1.4rem 1.6rem; text-align:center; margin-bottom:1rem; }
.verdict-likely-human { background: linear-gradient(135deg,#F0FDFA 0%,#CCFBF1 100%); border: 2px solid #14B8A6; border-radius:14px; padding:1.4rem 1.6rem; text-align:center; margin-bottom:1rem; }
.verdict-human { background: linear-gradient(135deg,#F0FDF4 0%,#DCFCE7 100%); border: 2px solid #22C55E; border-radius:14px; padding:1.4rem 1.6rem; text-align:center; margin-bottom:1rem; }
.verdict-uncertain { background: linear-gradient(135deg,#FFFBEB 0%,#FEF3C7 100%); border: 2px solid #F59E0B; border-radius:14px; padding:1.4rem 1.6rem; text-align:center; margin-bottom:1rem; }
.verdict-title-ai { font-size:1.7rem; font-weight:800; color:#B91C1C; }
.verdict-title-likely-ai { font-size:1.7rem; font-weight:800; color:#C2410C; }
.verdict-title-likely-human { font-size:1.7rem; font-weight:800; color:#0F766E; }
.verdict-title-human { font-size:1.7rem; font-weight:800; color:#15803D; }
.verdict-title-uncertain { font-size:1.7rem; font-weight:800; color:#92400E; }
.verdict-subtitle { font-size:1.05rem; margin-top:0.3rem; }

.text-box { background-color:#F8FAFC; border:1px solid #CBD5E1; border-radius:0.5rem; padding:1.2rem; font-family:'Inter',sans-serif; line-height:1.9; font-size:1.03rem; color:#1E293B; }
.sent-high-ai { background-color:#FEE2E2; color:#991B1B; font-weight:500; padding:0.15rem 0.35rem; border-radius:0.25rem; border-bottom:2px solid #EF4444; margin:0 0.05rem; }
.sent-low-ai { background-color:#DCFCE7; color:#166534; padding:0.15rem 0.35rem; border-radius:0.25rem; border-bottom:2px solid #22C55E; margin:0 0.05rem; }
.sent-short { background-color:#F1F5F9; color:#64748B; padding:0.15rem 0.35rem; border-radius:0.25rem; margin:0 0.05rem; }
.cliche-highlight { background-color:#FECACA; color:#991B1B; font-weight:600; padding:0.15rem 0.35rem; border-radius:0.25rem; border:1px solid #FCA5A5; }
.section-header { font-size:1.15rem; font-weight:700; color:#1E293B; border-left:4px solid #3B82F6; padding-left:0.75rem; margin:1.2rem 0 0.8rem 0; }
.section-header-audio { border-left-color:#8B5CF6; }

.cross-modal-consistent { background:linear-gradient(135deg,#F0FDF4 0%,#DCFCE7 100%); border:2px solid #22C55E; border-radius:14px; padding:1.4rem 1.6rem; margin:1rem 0; }
.cross-modal-conflict { background:linear-gradient(135deg,#FFFBEB 0%,#FEF3C7 100%); border:2px solid #F59E0B; border-radius:14px; padding:1.4rem 1.6rem; margin:1rem 0; }
.cross-modal-title-consistent { font-size:1.5rem; font-weight:800; color:#15803D; }
.cross-modal-title-conflict { font-size:1.5rem; font-weight:800; color:#B45309; }
.cross-modal-subtitle { font-size:1.1rem; font-weight:600; margin-top:0.3rem; color:#1E293B; }
.cross-modal-desc { font-size:0.95rem; margin-top:0.5rem; color:#334155; line-height:1.6; }

[data-testid="stSidebar"] { background:#0F172A; }
[data-testid="stSidebar"] * { color:#E2E8F0 !important; }
[data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown h3 { color:#F1F5F9 !important; }
</style>
"""
    st.markdown(analysis_css, unsafe_allow_html=True)

    # Back navigation
    nav_col1, nav_col2 = st.columns([1, 6])
    with nav_col1:
        if st.button("← Back to Home", key="btn_back_home", use_container_width=True):
            st.session_state["app_page"] = "landing"
            st.rerun()

    st.markdown('<div class="analysis-header-title">🔬 AI Text & Audio Forensics</div>', unsafe_allow_html=True)
    st.markdown('<div class="analysis-header-subtitle">Forensic detection of AI-generated text and synthetic voice deepfakes.</div>', unsafe_allow_html=True)

    # ── Mode selection — ONLY Text and Audio (no Combined) ────────────────────
    card_col1, card_col2 = st.columns(2)

    with card_col1:
        is_text_active = st.session_state["analysis_mode"] == "Text"
        active_cls = "mode-card mode-card-active" if is_text_active else "mode-card"
        st.markdown(
            f'<div class="{active_cls}">'
            f'<div class="mode-card-title">📝 TEXT FORENSICS</div>'
            f'<div class="mode-card-desc">Analyze written content for AI-associated linguistic patterns.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "✓ Selected: Text Forensics" if is_text_active else "Select Text Forensics",
            key="btn_select_text",
            type="primary" if is_text_active else "secondary",
            use_container_width=True,
        ):
            st.session_state["analysis_mode"] = "Text"
            st.rerun()

    with card_col2:
        is_audio_active = st.session_state["analysis_mode"] == "Audio"
        active_cls = "mode-card mode-card-active-audio" if is_audio_active else "mode-card"
        st.markdown(
            f'<div class="{active_cls}">'
            f'<div class="mode-card-title">🎙️ AUDIO FORENSICS</div>'
            f'<div class="mode-card-desc">Analyze speech for synthetic voice characteristics.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "✓ Selected: Audio Forensics" if is_audio_active else "Select Audio Forensics",
            key="btn_select_audio",
            type="primary" if is_audio_active else "secondary",
            use_container_width=True,
        ):
            st.session_state["analysis_mode"] = "Audio"
            st.rerun()

    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔬 AI Forensics System")
        st.markdown("---")
        mode_radio = st.radio(
            "**Analysis Mode**",
            options=["📝 Text Analysis", "🎙️ Audio Analysis"],
            index=0 if st.session_state["analysis_mode"] == "Text" else 1,
            key="sidebar_mode_radio",
        )
        current_selection = "Text" if "Text" in mode_radio else "Audio"
        if current_selection != st.session_state["analysis_mode"]:
            st.session_state["analysis_mode"] = current_selection
            st.rerun()

        st.markdown("---")
        if "Text" in mode_radio:
            st.markdown("### 📝 Text Forensics Pipeline")
            st.markdown("""
1. **Curvature** — Fast-DetectGPT via `distilgpt2`
2. **Burstiness** — Sentence-length σ/μ
3. **Lexical Entropy** — TTR + Shannon entropy
4. **Structural Regularity** — Starter diversity & POS overlap
5. **Cliché Scan** — 50+ overused AI buzzwords

Fused via calibrated 5-Feature Logistic Regression (F1 **94.95%** on multi-genre corpus).
""")
        else:
            st.markdown("### 🎙️ Audio Forensics Pipeline")
            st.markdown("""
1. **VAD** — Silero silence stripping
2. **ASR** — OpenAI Whisper-Tiny
3. **Deepfake Score** — `wav2vec2-deepfake-voice-detector`
$$S_{Audio} = \\text{Sigmoid}\\!\\left(\\frac{logit_{fake} - logit_{real}}{T}\\right) \\times 100$$

**Input Options:**
- 📁 **Upload Audio File** (.wav, .mp3, .flac, .ogg, .m4a, .aac)
- 🎤 **Live Recording** (Browser WebRTC capture)

**Stage 2:** Interactive Cross-Modality Analysis on Whisper transcript.
""")
        st.markdown("---")
        st.caption("CPU-only · PyTorch · Transformers · NLTK · spaCy · Silero VAD")

    mode = st.session_state["analysis_mode"]

    # ── Input Section ─────────────────────────────────────────────────────────
    st.subheader("1. Input")

    raw_text = ""
    uploaded_audio = None
    recorded_audio = None
    tmp_audio_path = None
    is_live_recording = False

    if mode == "Text":
        col_input, col_info = st.columns([3, 2])
        with col_input:
            st.markdown("**Provide text to analyze:**")
            input_mode = st.radio(
                "Input Method:",
                ["Paste Text", "Upload .txt"],
                horizontal=True,
                key="text_input_mode_selector",
            )
            if input_mode == "Paste Text":
                raw_text = st.text_area(
                    "Paste text to analyze:",
                    height=220,
                    placeholder="Paste text here for five-feature forensic analysis...",
                    key="text_area_input",
                )
            else:
                txt_file = st.file_uploader("Upload .txt file:", type=["txt"], key="txt_file_uploader")
                if txt_file:
                    try:
                        raw_text = txt_file.read().decode("utf-8")
                    except UnicodeDecodeError:
                        txt_file.seek(0)
                        raw_text = txt_file.read().decode("latin-1")

        with col_info:
            st.markdown("**Analysis Specifications:**")
            st.info(
                r"• **Engine**: Five-Feature Logistic Regression (DistilGPT-2 + Rhythm + Entropy + Regularity + Clichés)" "\n\n"
                r"• **Decision Tiers**: Human ($\le 36\%$) · Likely Human ($36\text{–}55\%$) · Likely AI ($55\text{–}75\%$) · AI ($\ge 75\%$)" "\n\n"
                r"• **Optimal Input**: Paragraphs $\ge 30$ words ($\ge 5$ sentences for full rhythm analysis)"
            )
            word_count = len(raw_text.split()) if raw_text else 0
            if raw_text:
                st.caption(f"📊 Text size: **{word_count} words** | **{len(raw_text)} characters**")
                if word_count <= 30:
                    st.error(
                        f"❌ **Text too short** — `{word_count}` word{'s' if word_count != 1 else ''} entered. "
                        "The Five-Feature pipeline requires **more than 30 words**."
                    )

        run_robustness = st.checkbox("Run adversarial robustness check (T5 paraphrase, adds ~30s)", value=False)
        _text_word_count = len(raw_text.split()) if raw_text else 0
        can_analyze = bool(raw_text.strip()) and _text_word_count > 30

    elif mode == "Audio":
        audio_source = st.radio(
            "**Audio Input Method:**",
            options=["📁 Upload Audio File", "🎤 Live Recording"],
            horizontal=True,
            key="audio_input_method_selector",
        )

        if audio_source == "📁 Upload Audio File":
            col_upload, col_preview = st.columns([1, 1])
            with col_upload:
                st.markdown("**Upload audio file to analyze:**")
                uploaded_audio = st.file_uploader(
                    "Supported formats: .wav, .mp3, .flac, .ogg, .m4a, .aac",
                    type=["wav", "mp3", "flac", "ogg", "m4a", "aac"],
                    key="audio_file_uploader",
                )
                if uploaded_audio:
                    st.info(f"**Filename:** `{uploaded_audio.name}`\n\n**File Size:** `{uploaded_audio.size / 1024:.1f} KB`")
                    current_file_id = getattr(uploaded_audio, "file_id", uploaded_audio.name)
                    if st.session_state["audio_file_id"] != current_file_id:
                        st.session_state["audio_file_id"] = current_file_id
                        for k in ["audio_result", "cross_modal_text_result", "cross_modal_audio_score", "cross_modal_audio_result", "cross_modal_reasoner_result"]:
                            st.session_state[k] = None

            with col_preview:
                st.markdown("**Audio Preview & Specs:**")
                if uploaded_audio:
                    st.audio(uploaded_audio, format=f"audio/{uploaded_audio.name.split('.')[-1]}")
                    st.caption("🔍 Pipeline: Silero VAD → Whisper-Tiny ASR → wav2vec2 Deepfake Classifier")
                else:
                    st.info("ℹ️ Upload an audio file to enable playback preview.")

            run_robustness = False
            recorded_audio = None
            tmp_audio_path = None
            can_analyze = uploaded_audio is not None

        else:
            is_live_recording = True

        if is_live_recording:
            col_live, col_live_info = st.columns([1, 1])
            with col_live:
                st.markdown("### 🎤 LIVE RECORDING")
                st.markdown(
                    "Record your voice and analyse it using the **Audio Forensics** pipeline.\n\n"
                    "Click the microphone button below to start recording."
                )
                recorded_audio = st.audio_input(label="Record voice from microphone:", key="live_recording_audio_input")

                if recorded_audio is not None:
                    rec_bytes = recorded_audio.getvalue()
                    rec_size_kb = len(rec_bytes) / 1024
                    if rec_size_kb > 0:
                        st.success(f"✅ **Recording complete.** Captured `{rec_size_kb:.1f} KB` audio.")
                        st.caption("Click **Analyse Recording** below to process with Audio Forensics.")
                        if st.button("🔄 Record Again", key="btn_record_again", use_container_width=True):
                            for k in ["audio_result", "cross_modal_text_result", "cross_modal_audio_score", "cross_modal_audio_result", "cross_modal_reasoner_result", "live_recording_audio_input"]:
                                if k in st.session_state:
                                    st.session_state[k] = None
                            st.rerun()
                    else:
                        st.error("❌ Recorded audio is empty. Please speak clearly and try again.")
                        recorded_audio = None

            with col_live_info:
                st.markdown("**🎙️ Live Recording Pipeline Specs:**")
                st.info(
                    "1. **Capture**: Browser-native `MediaRecorder` WebRTC microphone stream\n\n"
                    "2. **Audio Forensics**: Silero VAD silence stripping → Whisper-Tiny ASR → wav2vec2 classifier\n\n"
                    "3. **Transcribed Text**: Optional Five-Feature Logistic Regression analysis on generated transcript\n\n"
                    "4. **Cross-Modality**: Deterministic acoustic vs linguistic consistency evaluation"
                )
                st.caption("🔒 *Recordings are processed temporarily in memory and automatically deleted after the session.*")

            run_robustness = False
            uploaded_audio = None
            tmp_audio_path = None
            can_analyze = recorded_audio is not None and len(recorded_audio.getvalue()) > 0
        else:
            is_live_recording = False
            recorded_audio = None

    btn_label = "🚀 Analyze Text" if mode == "Text" else ("🚀 Analyse Recording" if is_live_recording else "🚀 Analyze Audio")
    analyze_btn = st.button(btn_label, type="primary", use_container_width=True, disabled=not can_analyze)

    # ── Pipeline Execution ────────────────────────────────────────────────────
    if analyze_btn:
        st.divider()
        st.subheader("2. Analysis Results")

        temp_dir = Path(_REPO_ROOT) / "temp"
        temp_dir.mkdir(exist_ok=True)

        if uploaded_audio is not None:
            file_ext = os.path.splitext(uploaded_audio.name)[1] or ".wav"
            tmp_audio_path = str(temp_dir / f"upload_{uuid.uuid4().hex[:10]}{file_ext}")
            with open(tmp_audio_path, "wb") as f:
                f.write(uploaded_audio.getvalue())
        elif recorded_audio is not None:
            tmp_audio_path = str(temp_dir / f"live_recording_{uuid.uuid4().hex[:10]}.wav")
            with open(tmp_audio_path, "wb") as f:
                f.write(recorded_audio.getvalue())

        t_start = time.time()
        progress_placeholder = st.empty()

        def _audio_progress(cur: int, total: int):
            if total > 0:
                progress_placeholder.info(f"🎙️ **Analyzing Audio Windows:** Window `{cur}` of `{total}` ({cur/total*100:.1f}%)")

        try:
            if mode == "Text" and raw_text.strip():
                with st.spinner("Running Five-Feature Text Forensics Pipeline..."):
                    result = run_full_pipeline(text=raw_text.strip(), audio_path=None, run_robustness=run_robustness)
            else:
                spinner_msg = "🎙️ Analysing Live Recording (VAD + Whisper-Tiny + wav2vec2)..." if is_live_recording else "Initializing Audio Forensics Pipeline..."
                with st.spinner(spinner_msg):
                    result = run_full_pipeline(text=None, audio_path=tmp_audio_path, batch_size=1, audio_progress_callback=_audio_progress)
                progress_placeholder.empty()
                st.session_state["audio_result"] = result
                for k in ["cross_modal_text_result", "cross_modal_reasoner_result", "cross_modal_audio_score", "cross_modal_audio_result"]:
                    st.session_state[k] = None

            elapsed = time.time() - t_start
        except Exception as e:
            progress_placeholder.empty()
            st.error(f"❌ **Pipeline Error**: `{type(e).__name__}`\n\n{str(e)}")
            if tmp_audio_path and os.path.exists(tmp_audio_path):
                try: os.remove(tmp_audio_path)
                except OSError: pass
            st.stop()
        finally:
            if tmp_audio_path and os.path.exists(tmp_audio_path):
                try: os.remove(tmp_audio_path)
                except OSError: pass

        if mode == "Text" and result.get("text_score") is not None and raw_text.strip():
            _render_text_panel(result, raw_text.strip())

            with st.expander("Advanced: Text Robustness Check"):
                stab = result.get("text_stability_flag", "skipped")
                if stab == "stable": st.success("✅ **Stable** — T5 paraphrase delta within threshold.")
                elif stab == "unstable": st.warning("⚠️ **Unstable** — Score is sensitive to phrasing changes.")
                elif stab == "skipped": st.info("ℹ️ Robustness check not run. Enable checkbox to run.")
                else: st.info(f"Status: `{stab}`")

            with st.expander("Advanced: Model Details & Legacy Baseline"):
                st.markdown("**Production Model Output:**")
                st.write(f"• **AI Probability**: `{result.get('text_ai_probability', result['text_score']/100):.4f}` ({result['text_score']:.2f}%)")
                st.write(f"• **Verdict**: `{result['text_verdict']}`")
                st.write(f"• **Model Used**: `{result.get('text_model_used', 'five_feature_logistic_regression')}`")
                st.divider()
                st.markdown("**Legacy Baseline Score (PATH A Reference):**")
                st.write(f"• **Legacy Score**: `{result.get('text_legacy_score', 'N/A')}` / 100")
                st.write(f"• **Signal Agreement**: `{result.get('text_signal_agreement', 'agreement')}`")
                st.caption("ℹ️ Legacy score uses historical Gaussian CDF heuristic fusion (audit trail only).")

            with st.expander("Show Complete Raw Output (JSON)"):
                st.json({
                    "production_ai_probability": result.get("text_ai_probability"),
                    "production_ai_score": result["text_score"],
                    "production_verdict": result["text_verdict"],
                    "model_used": result.get("text_model_used"),
                    "features": result.get("text_features"),
                    "legacy_baseline_score": result.get("text_legacy_score"),
                    "signals": result.get("text_signals"),
                })

        elif result.get("audio_score") is not None:
            _render_audio_panel(result)
            with st.expander("Show Raw Audio Signal Values (JSON)"):
                st.json({
                    "audio_score": result["audio_score"],
                    "audio_verdict": result["audio_verdict"],
                    "logit_fake": result["logit_fake"],
                    "logit_real": result["logit_real"],
                    "transcript_length": len(result.get("transcript", "")),
                    "audio_verdict_note": result["audio_verdict_note"],
                })

        st.divider()
        st.caption(f"⏱️ Total analysis time: **{elapsed:.2f} seconds**")

    else:
        # Persist audio results across reruns
        _persisted_audio = st.session_state.get("audio_result")
        if mode == "Audio" and _persisted_audio is not None and _persisted_audio.get("audio_score") is not None:
            st.divider()
            st.subheader("2. Analysis Results")
            _render_audio_panel(_persisted_audio)
            with st.expander("Show Raw Audio Signal Values (JSON)"):
                st.json({
                    "audio_score": _persisted_audio["audio_score"],
                    "audio_verdict": _persisted_audio["audio_verdict"],
                    "logit_fake": _persisted_audio["logit_fake"],
                    "logit_real": _persisted_audio["logit_real"],
                    "transcript_length": len(_persisted_audio.get("transcript", "")),
                    "audio_verdict_note": _persisted_audio.get("audio_verdict_note", ""),
                })
        else:
            if mode == "Text":
                st.info("💡 Paste or upload text above, then click **Analyze Text**.")
            elif is_live_recording:
                st.info("💡 Record audio above, then click **Analyse Recording**.")
            else:
                st.info("💡 Upload an audio clip above, then click **Analyze Audio**.")
