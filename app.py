"""
app.py  (repository root)
=========================
Streamlit Application — AI Text & Audio Forensics Detection System

Presents an interface for:
  - Text Analysis  (5-feature text_forensics pipeline)
  - Audio Analysis (audio_forensics pipeline)

Run from the repository root:
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from pathlib import Path

import nltk
import pandas as pd
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion_integration import run_full_pipeline

# ── Session-state initialisation ─────────────────────────────────────────────
# Persists audio result and cross-modal state across st.rerun() calls.
for _key in (
    "audio_result",         # full audio pipeline result dict
    "audio_file_id",        # uploaded_audio.file_id to detect new uploads
    "cross_modal_text_result",
    "cross_modal_audio_score",
    "cross_modal_audio_result",
    "cross_modal_reasoner_result",
):
    if _key not in st.session_state:
        st.session_state[_key] = None

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
# Page config + CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Text & Audio Forensics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Verdict cards ── */
.verdict-ai {
    background: linear-gradient(135deg, #FEF2F2 0%, #FFE4E6 100%);
    border: 2px solid #EF4444;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    margin-bottom: 1rem;
}
.verdict-likely-ai {
    background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%);
    border: 2px solid #F97316;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    margin-bottom: 1rem;
}
.verdict-likely-human {
    background: linear-gradient(135deg, #F0FDFA 0%, #CCFBF1 100%);
    border: 2px solid #14B8A6;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    margin-bottom: 1rem;
}
.verdict-human {
    background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
    border: 2px solid #22C55E;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    margin-bottom: 1rem;
}
.verdict-uncertain {
    background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
    border: 2px solid #F59E0B;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    margin-bottom: 1rem;
}
.verdict-title-ai            { font-size: 1.7rem; font-weight: 800; color: #B91C1C; }
.verdict-title-likely-ai     { font-size: 1.7rem; font-weight: 800; color: #C2410C; }
.verdict-title-likely-human  { font-size: 1.7rem; font-weight: 800; color: #0F766E; }
.verdict-title-human         { font-size: 1.7rem; font-weight: 800; color: #15803D; }
.verdict-title-uncertain     { font-size: 1.7rem; font-weight: 800; color: #92400E; }
.verdict-subtitle { font-size: 1.05rem; margin-top: 0.3rem; }

/* ── Text box ── */
.text-box {
    background-color: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 0.5rem;
    padding: 1.2rem;
    font-family: 'Inter', sans-serif;
    line-height: 1.9;
    font-size: 1.03rem;
    color: #1E293B;
}

/* ── Sentence highlights ── */
.sent-high-ai {
    background-color: #FEE2E2;
    color: #991B1B;
    font-weight: 500;
    padding: 0.15rem 0.35rem;
    border-radius: 0.25rem;
    border-bottom: 2px solid #EF4444;
    margin: 0 0.05rem;
}
.sent-low-ai {
    background-color: #DCFCE7;
    color: #166534;
    padding: 0.15rem 0.35rem;
    border-radius: 0.25rem;
    border-bottom: 2px solid #22C55E;
    margin: 0 0.05rem;
}
.sent-short {
    background-color: #F1F5F9;
    color: #64748B;
    padding: 0.15rem 0.35rem;
    border-radius: 0.25rem;
    margin: 0 0.05rem;
}
.cliche-highlight {
    background-color: #FECACA;
    color: #991B1B;
    font-weight: 600;
    padding: 0.15rem 0.35rem;
    border-radius: 0.25rem;
    border: 1px solid #FCA5A5;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1E293B;
    border-left: 4px solid #3B82F6;
    padding-left: 0.75rem;
    margin: 1.2rem 0 0.8rem 0;
}
.section-header-audio {
    border-left-color: #8B5CF6;
}

/* ── Uncalibrated badge ── */
.uncalibrated-badge {
    background: #FFF7ED;
    border: 1px solid #FB923C;
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
    font-size: 0.85rem;
    color: #7C2D12;
    margin-top: 0.5rem;
}

/* ── Cross-Modal Consistency & Conflict Badges ── */
.cross-modal-consistent {
    background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
    border: 2px solid #22C55E;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
}
.cross-modal-conflict {
    background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
    border: 2px solid #F59E0B;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
}
.cross-modal-title-consistent {
    font-size: 1.5rem;
    font-weight: 800;
    color: #15803D;
}
.cross-modal-title-conflict {
    font-size: 1.5rem;
    font-weight: 800;
    color: #B45309;
}
.cross-modal-subtitle {
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 0.3rem;
    color: #1E293B;
}
.cross-modal-desc {
    font-size: 0.95rem;
    margin-top: 0.5rem;
    color: #334155;
    line-height: 1.6;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0F172A; }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: #F1F5F9 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 AI Forensics System")
    st.markdown("---")
    mode = st.radio(
        "**Analysis Mode**",
        options=["📝 Text Analysis", "🎙️ Audio Analysis", "🎤 Live Microphone"],
        index=0,
    )
    st.markdown("---")

    if "Text" in mode:
        st.markdown("### 📝 Text Forensics Pipeline")
        st.markdown("""
        1. **Curvature** — Fast-DetectGPT via `distilgpt2`
        2. **Burstiness** — Sentence-length σ/μ
        3. **Lexical Entropy** — TTR + Shannon entropy
        4. **Structural Regularity** — Starter diversity & POS overlap
        5. **Cliché Scan** — 50+ overused AI buzzwords

        Fused via calibrated 5-Feature Logistic Regression (F1 **94.95%** on multi-genre corpus).
        """)

    elif "Audio" in mode:
        st.markdown("### 🎙️ Audio Forensics Pipeline")
        st.markdown("""
        1. **VAD** — Silero silence stripping
        2. **ASR** — OpenAI Whisper-Tiny
        3. **Deepfake Score** — `wav2vec2-deepfake-voice-detector`
        $$S_{Audio} = \\text{Sigmoid}\\!\\left(\\frac{logit_{fake} - logit_{real}}{T}\\right) \\times 100$$
        """)

    elif "Live" in mode:
        st.markdown("### 🎤 Live Microphone Mode")
        st.markdown("""
        Record directly from your microphone.

        **Pipeline:**
        1. **Record** — Browser-native audio capture
        2. **VAD** — Silero silence stripping
        3. **ASR** — Whisper-Tiny transcription
        4. **Deepfake Score** — wav2vec2 audio classifier
        5. **Text Forensics** — Five-Feature LR on transcript
        6. **Cross-Modality** — Combined audio + text verdict
        """)

    st.markdown("---")
    st.caption("CPU-only · PyTorch · Transformers · NLTK · spaCy · Silero VAD")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 style="font-size:2.2rem;font-weight:800;color:#1E293B;margin-bottom:0.1rem;">🔬 AI Text & Audio Forensics</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#64748B;font-size:1.05rem;margin-bottom:1.5rem;">Forensic detection of AI-generated text and synthetic voice deepfakes.</p>', unsafe_allow_html=True)


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
# Helper: render text results panel (Section 3A)
# ─────────────────────────────────────────────────────────────────────────────
def _render_text_panel(result: dict, raw_text: str) -> None:
    text_score = result["text_score"]
    text_verdict = result["text_verdict"]
    model_name = result.get("text_model_used", "Five-Feature Logistic Regression")
    if "five_feature" in model_name:
        model_display = "Five-Feature Logistic Regression"
    else:
        model_display = "Corrected Four-Feature Baseline"

    st.markdown('<div class="section-header">📝 Text Forensics Results</div>', unsafe_allow_html=True)
    st.markdown(_verdict_card(text_verdict, text_score, model_name=model_display), unsafe_allow_html=True)

    if result.get("text_signal_agreement") == "disagreement":
        st.error(
            "⚠️ **Signals Disagree Significantly**: Individual detectors are giving "
            "conflicting evidence (sub-score spread > 40 points). Review the feature breakdown below."
        )

    st.divider()

    # Sentence-level highlighting
    st.markdown("**Sentence-Level Evidence** *(red = AI-likely, green = human-likely)*")
    baseline = _load_baseline_stats() if _TEXT_EXTRAS_AVAILABLE else None
    _render_sentence_highlights(raw_text, baseline)

    st.divider()

    # Core Features + clichés in tabs
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
        df = pd.DataFrame(feature_rows)
        st.dataframe(df, use_container_width=True)
        st.caption("ℹ️ *Note: These are raw forensic feature values fed into the Logistic Regression model, not individual AI probabilities.*")

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
# Helper: render audio results panel (Section 3B)
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
    st.markdown(
        _verdict_card(audio_verdict, audio_score, model_name="wav2vec2-deepfake-voice-detector"),
        unsafe_allow_html=True,
    )

    st.divider()

    # Logit metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Deepfake Score (Max)", f"{audio_score:.2f}%")
    col2.metric("Logit (Fake)", f"{logit_fake:.4f}")
    col3.metric("Logit (Real)", f"{logit_real:.4f}")
    col4.metric("Δ (Fake−Real)", f"{logit_diff:+.4f}")

    st.markdown("**Synthetic Risk Bar:**")
    st.progress(min(max(audio_score / 100.0, 0.0), 1.0))

    # Extended window metrics
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
        f"> **Decision Threshold:** `{threshold:.1f}%` | **Aggregation Strategy:** `max_risk` | **Windows:** `{n_windows}`"
    )

    # Audio metadata & specs card
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
            f"⚙️ **Configuration:** Window Size = `{win_sz}s` | Overlap = `{win_ov}s` | Step = `{win_step}s` | "
            f"Sampling Rate = `{audio_meta.get('normalized_sample_rate', 16000)} Hz mono`"
        )
        if n_windows > 1:
            st.caption(f"🔍 **Per-window scores ({n_windows} windows):** {[f'{s:.1f}%' for s in window_scores]}")

    st.divider()

    # ── Process Timing Breakdown ──────────────────────────────────────────────
    vad_time = result.get("audio_vad_time")
    asr_time = result.get("audio_asr_time")
    deepfake_time = result.get("audio_deepfake_time")
    total_time = result.get("audio_total_time")

    if vad_time is not None or asr_time is not None or deepfake_time is not None:
        st.markdown("**Process Timing Breakdown**")
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        if vad_time is not None:
            t_col1.metric("Silero VAD", f"{vad_time:.2f} s")
        if asr_time is not None:
            t_col2.metric("Whisper ASR", f"{asr_time:.2f} s")
        if deepfake_time is not None:
            t_col3.metric("Deepfake Score", f"{deepfake_time:.2f} s")
        if total_time is not None:
            t_col4.metric("Total Pipeline", f"{total_time:.2f} s")
        st.divider()

    # Transcript Section with Interactive Action
    st.markdown("**Speech-to-Text Transcript (Whisper-Tiny ASR)**")
    has_transcript = bool(transcript and transcript.strip())
    transcript_word_count = len(transcript.split()) if has_transcript else 0
    transcript_long_enough = transcript_word_count > 30

    if has_transcript:
        st.text_area("Transcribed Audio Content", value=transcript, height=110, disabled=True)
        st.caption(f"📊 {transcript_word_count} words | {len(transcript)} characters")
    else:
        st.warning("⚠️ No spoken text transcribed (silence, noise, or non-speech audio).")

    # ── Interactive Trigger for Stage 2: Cross-Modality Analysis ──────────────
    st.markdown("---")
    st.markdown('<div class="section-header section-header-audio">🧬 Stage 2 — Cross-Modality Analysis</div>', unsafe_allow_html=True)

    if not has_transcript:
        st.info("ℹ️ A valid speech transcript is required to perform Cross-Modality Analysis.")
    elif not transcript_long_enough:
        st.error(
            f"❌ **Transcript too short to analyse** — `{transcript_word_count}` word{'s' if transcript_word_count != 1 else ''} transcribed. "
            "The Five-Feature Text pipeline requires **more than 30 words** to produce reliable results. "
            "The transcribed audio segment is too brief for cross-modality text analysis."
        )
    else:
        # Check if text analysis for this session transcript was already run
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

        # ── Render Cross-Modality Results if available ────────────────────────
        cm_reasoner = st.session_state.get("cross_modal_reasoner_result")
        cm_text_res = st.session_state.get("cross_modal_text_result")

        if cm_reasoner and cm_text_res and st.session_state.get("cross_modal_audio_score") == audio_score:
            # 1. Text Analysis Details
            st.markdown("#### 📝 Transcribed Text Forensic Evaluation")
            txt_score = cm_reasoner["text"]["score"]
            txt_thresh = cm_reasoner["text"]["threshold"]
            txt_verdict = cm_reasoner["text"]["verdict"]
            txt_class = cm_reasoner["text"]["classification"]

            t_c1, t_c2, t_c3 = st.columns(3)
            t_c1.metric("Text AI Score", f"{txt_score:.2f}%")
            t_c2.metric("Decision Threshold", f"{txt_thresh:.1f}%")
            t_c3.metric("Text Classification", f"{txt_class} ({txt_verdict})")

            # 2. Cross-Modality Reasoner Comparison Card
            box_class = "cross-modal-consistent" if cm_reasoner["consistent"] else "cross-modal-conflict"
            title_class = "cross-modal-title-consistent" if cm_reasoner["consistent"] else "cross-modal-title-conflict"

            st.markdown(
                f"""
                <div class="{box_class}">
                    <div class="{title_class}">{cm_reasoner['title']}</div>
                    <div class="cross-modal-subtitle">{cm_reasoner['classification']}</div>
                    <div class="cross-modal-desc">{cm_reasoner['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 3. Evidence Breakdown
            st.markdown("#### 🔍 Cross-Modality Evidence Comparison")
            ev_col1, ev_col2 = st.columns(2)
            with ev_col1:
                st.markdown("**🎙️ Audio Modality**")
                st.write(f"• **Score:** `{cm_reasoner['audio']['score']:.2f}%`")
                st.write(f"• **Decision Threshold:** `{cm_reasoner['audio']['threshold']:.1f}%`")
                st.write(f"• **Classification:** `{cm_reasoner['audio']['classification']}` ({cm_reasoner['audio']['verdict']})")
                st.write(f"• **Model:** `wav2vec2-deepfake-voice-detector`")

            with ev_col2:
                st.markdown("**📝 Text Modality (Transcript)**")
                st.write(f"• **Score:** `{cm_reasoner['text']['score']:.2f}%`")
                st.write(f"• **Decision Threshold:** `{cm_reasoner['text']['threshold']:.1f}%`")
                st.write(f"• **Classification:** `{cm_reasoner['text']['classification']}` ({cm_reasoner['text']['verdict']})")
                st.write(f"• **Model:** `Five-Feature Logistic Regression`")
        elif not transcribed_text_analyzed:
            st.info("💡 Click **Analyse Transcribed Text** above to evaluate the transcript and compare cross-modal evidence.")

    st.divider()

    # Confidence explanation
    abs_diff = abs(logit_diff)
    extreme_tier  = conf_tiers.get("extreme_logit_margin", 3.0)
    high_tier     = conf_tiers.get("high_logit_margin", 1.5)
    moderate_tier = conf_tiers.get("moderate_logit_margin", 0.5)
    if abs_diff > extreme_tier:
        confidence = "Extreme Confidence"
    elif abs_diff > high_tier:
        confidence = "High Confidence"
    elif abs_diff > moderate_tier:
        confidence = "Moderate Confidence"
    else:
        confidence = "Borderline / Low Confidence"

    with st.expander("📋 Audio Classification Explanation"):
        if audio_score >= 75.0:
            tier_label = "Authentic AI Voice"
            tier_range = "75–100%"
        elif audio_score >= 56.0:
            tier_label = "Likely AI Voice"
            tier_range = "56–74%"
        elif audio_score >= 36.0:
            tier_label = "Likely Human Voice"
            tier_range = "36–55%"
        else:
            tier_label = "Authentic Human Voice"
            tier_range = "0–35%"

        if audio_score >= 56.0:
            st.markdown(
                f"**Why classified as {tier_label}?**\n\n"
                f"- **Tier:** Score `{audio_score:.2f}%` falls in the **{tier_label}** range ({tier_range}).\n"
                f"- **Logit Dominance**: `logit_fake` ({logit_fake:.4f}) > `logit_real` "
                f"({logit_real:.4f}) by **{logit_diff:+.4f}**.\n"
                f"- **Sigmoid Mapping**: Yields deepfake risk score of **{audio_score:.2f}%** (≥{threshold:.1f}% decision threshold).\n"
                f"- **Confidence**: **{confidence}** (|Δ| = `{abs_diff:.4f}`).\n"
                f"- The wav2vec2 model detected spectral artifacts or phase mismatches typical of synthetic speech."
            )
        else:
            st.markdown(
                f"**Why classified as {tier_label}?**\n\n"
                f"- **Tier:** Score `{audio_score:.2f}%` falls in the **{tier_label}** range ({tier_range}).\n"
                f"- **Logit Dominance**: `logit_real` ({logit_real:.4f}) ≥ `logit_fake` "
                f"({logit_fake:.4f}); margin: **{logit_diff:+.4f}**.\n"
                f"- **Sigmoid Mapping**: Deepfake risk score is **{audio_score:.2f}%** (<{threshold:.1f}% decision threshold).\n"
                f"- **Confidence**: **{confidence}** (|Δ| = `{abs_diff:.4f}`).\n"
                f"- Acoustic features align with natural human vocal tract resonances."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main App: Section 1 — Inputs
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("1. Input")

raw_text = ""
uploaded_audio = None
tmp_audio_path = None

if "Text" in mode:
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
            txt_file = st.file_uploader(
                "Upload .txt file:",
                type=["txt"],
                key="txt_file_uploader",
            )
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
            r"• **Decision Tiers**: Human ($\le 20\%$) · Likely Human ($20\text{–}45\%$) · Likely AI ($45\text{–}70\%$) · AI ($\ge 70\%$)" "\n\n"
            r"• **Optimal Input**: Paragraphs $\ge 30$ words ($\ge 5$ sentences for full rhythm analysis)"
        )
        word_count = len(raw_text.split()) if raw_text else 0
        if raw_text:
            st.caption(f"📊 Text size: **{word_count} words** | **{len(raw_text)} characters**")
            if word_count <= 30:
                st.error(
                    f"❌ **Text too short to analyse** — `{word_count}` word{'s' if word_count != 1 else ''} entered. "
                    "The Five-Feature pipeline requires **more than 30 words** to compute "
                    "burstiness, structural regularity, and entropy features reliably. "
                    "Please provide a longer passage."
                )

    run_robustness = st.checkbox("Run adversarial robustness check (T5 paraphrase, adds ~30s)", value=False)
    # Gate: must have text AND more than 30 words
    _text_word_count = len(raw_text.split()) if raw_text else 0
    can_analyze = bool(raw_text.strip()) and _text_word_count > 30

elif "Audio" in mode:
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
                st.session_state["audio_file_id"]              = current_file_id
                st.session_state["audio_result"]               = None
                st.session_state["cross_modal_text_result"]    = None
                st.session_state["cross_modal_audio_score"]    = None
                st.session_state["cross_modal_audio_result"]   = None
                st.session_state["cross_modal_reasoner_result"]= None

    with col_preview:
        st.markdown("**Audio Preview & Specs:**")
        if uploaded_audio:
            st.audio(uploaded_audio, format=f"audio/{uploaded_audio.name.split('.')[-1]}")
            st.caption("🔍 Pipeline: Silero VAD $\\to$ Whisper-Tiny ASR $\\to$ wav2vec2 Deepfake Classifier")
        else:
            st.info("ℹ️ Upload an audio file to enable playback preview and forensic scoring.")

    run_robustness = False
    mic_audio = None
    tmp_audio_path = None
    can_analyze = uploaded_audio is not None

else:  # Live Microphone
    col_mic, col_mic_info = st.columns([1, 1])
    with col_mic:
        st.markdown("**Record from your microphone:**")
        try:
            from streamlit_mic_recorder import mic_recorder
            mic_audio = mic_recorder(
                start_prompt="⏺ Start Recording",
                stop_prompt="⏹ Stop Recording",
                just_once=True,
                key="live_mic_recorder",
            )
        except ImportError:
            st.error("❌ `streamlit-mic-recorder` not installed.")
            mic_audio = None

    with col_mic_info:
        st.markdown("**Live Capture Specs:**")
        st.info(
            "🎤 **Capture**: Browser-native WebRTC microphone input\n\n"
            "🔍 **Pipeline**: VAD → Whisper-Tiny → wav2vec2 → Text LR → Cross-Modality\n\n"
            "📊 **Output**: Audio score + Text score + Combined verdict"
        )
        if mic_audio:
            st.caption(f'🎙️ Recording captured · {len(mic_audio["bytes"]) // 1024} KB')
            st.audio(mic_audio['bytes'], format="audio/wav")

    run_robustness = False
    uploaded_audio = None
    tmp_audio_path = None
    if mic_audio:
        import wave
        _mic_bytes = mic_audio['bytes']
        _mic_sr    = mic_audio.get('sample_rate', 16000)
        _mic_sw    = mic_audio.get('sample_width', 2)
        _mic_ch    = mic_audio.get('num_channels', 1)
        tmp_mic = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(tmp_mic.name, 'wb') as wf:
            wf.setnchannels(_mic_ch)
            wf.setsampwidth(_mic_sw)
            wf.setframerate(_mic_sr)
            wf.writeframes(_mic_bytes)
        tmp_audio_path = tmp_mic.name
        if st.session_state.get("mic_audio_id") != id(_mic_bytes):
            st.session_state["mic_audio_id"]               = id(_mic_bytes)
            st.session_state["audio_result"]               = None
            st.session_state["cross_modal_text_result"]    = None
            st.session_state["cross_modal_audio_score"]    = None
            st.session_state["cross_modal_audio_result"]   = None
            st.session_state["cross_modal_reasoner_result"]= None
    can_analyze = mic_audio is not None

analyze_btn = st.button(
    "🚀 Analyze Text" if "Text" in mode else ("🚀 Analyze Audio" if "Audio" in mode else "🚀 Analyze Live Recording"),
    type="primary",
    use_container_width=True,
    disabled=not can_analyze,
)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Pipeline Execution
# ─────────────────────────────────────────────────────────────────────────────
if analyze_btn:
    st.divider()
    st.subheader("2. Analysis Results")

    # Save audio to temp file if provided
    if uploaded_audio is not None:
        file_ext = os.path.splitext(uploaded_audio.name)[1] or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        tmp.write(uploaded_audio.getvalue())
        tmp.close()
        tmp_audio_path = tmp.name

    t_start = time.time()

    progress_placeholder = st.empty()

    def _audio_progress(cur: int, total: int):
        if total > 0:
            pct = cur / total
            progress_placeholder.info(f"🎙️ **Analyzing Audio Windows:** Window `{cur}` of `{total}` ({pct*100:.1f}%)")

    try:
        if "Text" in mode and raw_text.strip():
            with st.spinner("Running Five-Feature Text Forensics Pipeline..."):
                result = run_full_pipeline(
                    text=raw_text.strip(),
                    audio_path=None,
                    run_robustness=run_robustness,
                )

        elif "Live" in mode:
            # ── Live Mic: Full combined pipeline (Audio + Text + Cross-Modal) ──
            with st.spinner("🎙️ Running Audio Forensics (VAD + Whisper-Tiny + wav2vec2)..."):
                result = run_full_pipeline(
                    text=None,
                    audio_path=tmp_audio_path,
                    batch_size=1,
                    audio_progress_callback=_audio_progress,
                )
            progress_placeholder.empty()
            st.session_state["audio_result"] = result

            # Auto-run text analysis on the transcript (combined mode)
            transcript_for_text = (result.get("transcript") or "").strip()
            transcript_wc = len(transcript_for_text.split()) if transcript_for_text else 0
            if transcript_wc > 30:
                with st.spinner("📝 Running Text Forensics on live transcript..."):
                    from text_forensics.pipeline import analyze_text
                    from cross_modal_reasoner import evaluate_cross_modality
                    text_res = analyze_text(transcript_for_text, run_robustness=False)
                    reasoner_res = evaluate_cross_modality(result, text_res)
                    st.session_state["cross_modal_text_result"]     = text_res
                    st.session_state["cross_modal_audio_score"]     = result.get("audio_score")
                    st.session_state["cross_modal_audio_result"]    = result
                    st.session_state["cross_modal_reasoner_result"] = reasoner_res

        else:
            with st.spinner("Initializing Audio Forensics Pipeline (VAD + ASR + wav2vec2)..."):
                result = run_full_pipeline(
                    text=None,
                    audio_path=tmp_audio_path,
                    batch_size=1,
                    audio_progress_callback=_audio_progress,
                )
            progress_placeholder.empty()
            # Persist audio result so cross-modal button reruns can access it
            st.session_state["audio_result"] = result

        elapsed = time.time() - t_start
    except Exception as e:
        progress_placeholder.empty()
        err_msg = str(e)
        st.error(f"❌ **Pipeline Error**: `{type(e).__name__}`\n\n{err_msg}")
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        st.stop()
    finally:
        # Cleanup temp audio file
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            try:
                os.remove(tmp_audio_path)
            except OSError:
                pass

    # ── Section 3A: Text panel ─────────────────────────────────────────────────
    if "Text" in mode and result.get("text_score") is not None and raw_text.strip():
        _render_text_panel(result, raw_text.strip())

        with st.expander("Advanced: Text Robustness Check"):
            stab = result.get("text_stability_flag", "skipped")
            if stab == "stable":
                st.success("✅ **Stable** — T5 paraphrase delta within threshold.")
            elif stab == "unstable":
                st.warning("⚠️ **Unstable** — Score is sensitive to phrasing changes.")
            elif stab == "skipped":
                st.info("ℹ️ Robustness check not run. Enable checkbox to run.")
            else:
                st.info(f"Status: `{stab}`")

        with st.expander("Advanced: Model Details & Legacy Baseline"):
            st.markdown("**Production Model Output:**")
            st.write(f"• **AI Probability**: `{result.get('text_ai_probability', result['text_score']/100):.4f}` ({result['text_score']:.2f}%)")
            st.write(f"• **Verdict**: `{result['text_verdict']}`")
            st.write(f"• **Model Used**: `{result.get('text_model_used', 'five_feature_logistic_regression')}`")
            st.divider()
            st.markdown("**Legacy Baseline Score (PATH A Reference):**")
            st.write(f"• **Legacy Score**: `{result.get('text_legacy_score', 'N/A')}` / 100")
            st.write(f"• **Signal Agreement**: `{result.get('text_signal_agreement', 'agreement')}`")
            st.caption("ℹ️ The legacy score uses historical Gaussian CDF heuristic fusion and is preserved only for audit trail/backward compatibility.")

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

    # ── Section 3B: Audio panel ────────────────────────────────────────────────
    elif "Audio" in mode and result.get("audio_score") is not None:
        _render_audio_panel(result)

        with st.expander("Show Raw Audio Signal Values (JSON)"):
            st.json({
                "audio_score": result["audio_score"],
                "audio_verdict": result["audio_verdict"],
                "logit_fake": result["logit_fake"],
                "logit_real": result["logit_real"],
                "transcript_length": len(result["transcript"]),
                "audio_verdict_note": result["audio_verdict_note"],
            })

    # -- Section 3C: Live Mic combined panel --
    elif "Live" in mode and result.get("audio_score") is not None:
        _render_audio_panel(result)

        # Show auto-rendered cross-modal result if transcript was long enough
        cm_r = st.session_state.get("cross_modal_reasoner_result")
        cm_t = st.session_state.get("cross_modal_text_result")
        _live_transcript = (result.get("transcript") or "").strip()
        _live_wc = len(_live_transcript.split()) if _live_transcript else 0
        if _live_wc <= 30:
            st.warning(
                f"⚠️ Transcript too short ({_live_wc} words) for text forensics — "
                "audio deepfake score shown above is still valid."
            )
        elif cm_r and cm_t:
            st.markdown("---")
            st.markdown(
                '<div class="section-header section-header-audio">🧬 Combined Analysis — Cross-Modality Verdict</div>',
                unsafe_allow_html=True,
            )
            _cls = cm_r["classification"]
            _consistent = cm_r["consistent"]
            _card_cls = "cross-modal-consistent" if _consistent else "cross-modal-conflict"
            _icon = "✅" if _consistent else "⚠️"
            _label = "CONSISTENT" if _consistent else "CONFLICT"
            st.markdown(
                f'<div class="cross-modal-card {_card_cls}">'
                f'<div class="cross-modal-badge">{_icon} {_label}</div>'
                f'<div class="cross-modal-title">{_cls.replace("_", " ")}</div>'
                f'<div class="cross-modal-desc">{cm_r.get("description", "")}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🎙️ Audio Modality**")
                st.write(f"• **Score:** `{cm_r['audio']['score']:.2f}%`")
                st.write(f"• **Threshold:** `{cm_r['audio']['threshold']:.1f}%`")
                st.write(f"• **Classification:** `{cm_r['audio']['classification']}` ({cm_r['audio']['verdict']})")
            with c2:
                st.markdown("**📝 Text Modality (Transcript)**")
                st.write(f"• **Score:** `{cm_r['text']['score']:.2f}%`")
                st.write(f"• **Threshold:** `{cm_r['text']['threshold']:.1f}%`")
                st.write(f"• **Classification:** `{cm_r['text']['classification']}` ({cm_r['text']['verdict']})")

        with st.expander("Show Raw Signal Values (JSON)"):
            st.json({
                "audio_score": result["audio_score"],
                "audio_verdict": result["audio_verdict"],
                "logit_fake": result["logit_fake"],
                "logit_real": result["logit_real"],
                "transcript_word_count": _live_wc,
                "cross_modal_classification": cm_r["classification"] if cm_r else "N/A",
            })

    st.divider()
    st.caption(f"⏱️ Total analysis time: **{elapsed:.2f} seconds**")

else:
    # ── Show persisted audio results + cross-modal panel on reruns ─────────────
    # When analyse_btn was NOT clicked (e.g. after st.rerun() from cross-modal
    # button), restore from session state so the panel and Stage 2 remain visible.
    _persisted_audio = st.session_state.get("audio_result")
    if ("Audio" in mode or "Live" in mode) and _persisted_audio is not None and _persisted_audio.get("audio_score") is not None:
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
                "audio_verdict_note": _persisted_audio["audio_verdict_note"],
            })
    else:
        # Empty state
        if "Text" in mode:
            st.info("💡 Paste or upload text above, then click **Analyze Text**.")
        elif "Live" in mode:
            st.info("💡 Record audio above, then click **Analyze Live Recording**.")
        else:
            st.info("💡 Upload an audio clip above, then click **Analyze Audio**.")
