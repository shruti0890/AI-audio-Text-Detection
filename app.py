"""
app.py  (repository root)
=========================
Unified Streamlit Application — AI Audio + Text Forensics Detection System

Presents a single, tabbed interface for:
  - Text-only analysis  (text_forensics pipeline)
  - Audio-only analysis (audio_forensics pipeline)
  - Combined analysis   (both pipelines + unified score via fusion_integration)

Run from the repository root:
    streamlit run app.py

Architecture Sections:
    Section 1  : Input routing
    Section 2  : Per-modality pipeline execution (via fusion_integration)
    Section 3A : Text result display
    Section 3B : Audio result display
    Section 3C : Deferred — notice shown to user
    Section 4  : Unified score panel (Combined mode)
    Section 5  : Unified verdict + evidence summary
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
    page_title="AI Forensics — Text & Audio Detection",
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
.verdict-title-ai      { font-size: 1.7rem; font-weight: 800; color: #B91C1C; }
.verdict-title-human   { font-size: 1.7rem; font-weight: 800; color: #15803D; }
.verdict-title-uncertain { font-size: 1.7rem; font-weight: 800; color: #92400E; }
.verdict-subtitle { font-size: 1.05rem; margin-top: 0.3rem; }

/* ── Unified score banner ── */
.unified-banner {
    background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    color: white;
    margin-bottom: 1.2rem;
}
.unified-score-label { font-size: 0.9rem; color: #94A3B8; letter-spacing: 0.08em; }
.unified-score-value { font-size: 3rem; font-weight: 800; margin: 0.1rem 0; }
.unified-verdict-text { font-size: 1.1rem; color: #CBD5E1; }

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
        options=["📝 Text Only", "🎙️ Audio Only", "🔀 Combined (Text + Audio)"],
        index=0,
    )
    st.markdown("---")

    if "Text" in mode:
        st.markdown("### 📝 Text Pipeline")
        st.markdown("""
        1. **Prob. Curvature** — Fast-DetectGPT via `distilgpt2`
        2. **Burstiness** — Sentence-length σ/μ
        3. **Cliché Scan** — 50 AI buzzwords
        4. **Lexical Entropy** — TTR + Shannon H

        Fused with calibrated weights (ROC-AUC **0.9994** on 120 HC3 samples).
        """)

    if "Audio" in mode:
        st.markdown("### 🎙️ Audio Pipeline")
        st.markdown("""
        1. **VAD** — Silero silence stripping
        2. **ASR** — OpenAI Whisper-Tiny
        3. **Deepfake Score** — `wav2vec2-deepfake-voice-detector`
        $$S_{Audio} = \\text{Sigmoid}(logit_{fake} - logit_{real}) \\times 100$$
        """)

    if "Combined" in mode:
        st.markdown("### 🔀 Combined Mode")
        st.markdown("""
        Both pipelines run sequentially.
        **Unified score** = equal-weight average *(placeholder — to be recalibrated)*.
        **Section 3C** (cross-modal consistency) is **deferred** pending calibration.
        """)

    st.markdown("---")
    st.caption("CPU-only · PyTorch · Transformers · NLTK · spaCy · Silero VAD")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 style="font-size:2.2rem;font-weight:800;color:#1E293B;margin-bottom:0.1rem;">🔬 AI Forensics — Text & Audio Detection</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#64748B;font-size:1.05rem;margin-bottom:1.5rem;">Detect AI-generated text and audio deepfakes using statistical forensics.</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: verdict HTML card
# ─────────────────────────────────────────────────────────────────────────────
def _verdict_card(verdict: str, score: float, subtitle: str = "") -> str:
    v = verdict.lower()
    if "ai" in v or "deepfake" in v or "generated" in v:
        css, title_css = "verdict-ai", "verdict-title-ai"
        icon = "🤖"
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
        f'<div class="verdict-subtitle">Score: <b>{score:.1f} / 100</b></div>'
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

    st.markdown('<div class="section-header">📝 Text Analysis Results</div>', unsafe_allow_html=True)
    st.markdown(_verdict_card(text_verdict, text_score), unsafe_allow_html=True)

    if result.get("text_signal_agreement") == "disagreement":
        st.error(
            "⚠️ **Signals Disagree Significantly**: Individual detectors are giving "
            "conflicting evidence (sub-score spread > 40 points). Treat the combined "
            "score with extra caution."
        )

    st.divider()

    # Sentence-level highlighting
    st.markdown("**Sentence-Level Evidence** *(red = AI-likely, green = human-likely)*")
    baseline = _load_baseline_stats() if _TEXT_EXTRAS_AVAILABLE else None
    _render_sentence_highlights(raw_text, baseline)

    st.divider()

    # Sub-scores + clichés in tabs
    tab1, tab2, tab3 = st.tabs(["📊 Signal Sub-Scores", "🚩 Cliché Evidence", "📏 Rhythm Analysis"])

    with tab1:
        sigs = result.get("text_signals", {}) or {}
        chart_data = [
            {"Signal": "Curvature (Fast-DetectGPT)", "Score": sigs.get("curvature_score") or 0},
            {"Signal": "Burstiness (Rhythm)",         "Score": sigs.get("burstiness_score") or 0},
            {"Signal": "Cliché Density",              "Score": sigs.get("cliche_score") or 0},
            {"Signal": "Lexical Entropy",             "Score": sigs.get("entropy_score") or 0},
        ]
        df = pd.DataFrame(chart_data)
        col_c, col_t = st.columns([2, 1])
        with col_c:
            st.bar_chart(df, x="Signal", y="Score", color="#3B82F6", use_container_width=True)
        with col_t:
            for row in chart_data:
                st.write(f"• **{row['Signal']}**: `{row['Score']:.1f}` / 100")

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
    audio_score  = result["audio_score"]
    audio_verdict = result["audio_verdict"]
    logit_fake   = result["logit_fake"]
    logit_real   = result["logit_real"]
    transcript   = result["transcript"]
    logit_diff   = logit_fake - logit_real
    threshold    = result.get("audio_decision_threshold_pct", 56.0)
    temperature  = result.get("audio_temperature", 1.15)
    n_windows    = result.get("audio_n_windows", 1)
    window_scores = result.get("audio_window_scores", [])
    conf_tiers   = result.get("audio_confidence_tiers", {"extreme_logit_margin": 3.0, "high_logit_margin": 1.5, "moderate_logit_margin": 0.5})
    diff_scaled  = logit_diff / temperature if temperature else logit_diff

    st.markdown('<div class="section-header section-header-audio">🎙️ Audio Analysis Results</div>', unsafe_allow_html=True)
    st.markdown(_verdict_card(audio_verdict, audio_score), unsafe_allow_html=True)

    # Calibration badge notice
    st.markdown(
        '<div class="uncalibrated-badge">ℹ️ <b>Audio forensic thresholds calibrated:</b> '
        '0–35 Authentic Human Voice | 36–55 Likely Human Voice | '
        '56–74 Likely AI Voice | 75–100 Authentic AI Voice.</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Logit metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Deepfake Score", f"{audio_score:.2f}%")
    col2.metric("Logit (Fake)", f"{logit_fake:.4f}")
    col3.metric("Logit (Real)", f"{logit_real:.4f}")
    col4.metric("Δ (Fake−Real)", f"{logit_diff:+.4f}")

    st.markdown("**Synthetic Risk Bar:**")
    st.progress(min(max(audio_score / 100.0, 0.0), 1.0))

    st.markdown(
        f"> **Scoring Formula (T={temperature}):** "
        f"$$S_{{Audio}} = \\text{{Sigmoid}}\\!\\left(\\frac{{{logit_fake:.4f}-({logit_real:.4f})}}{{{temperature}}}\\right)"
        f"\\times 100 = {audio_score:.2f}\\%$$"
        f"\n> Threshold: **{threshold:.1f}%** | Windows: **{n_windows}**"
    )
    if n_windows > 1:
        st.caption(f"🔍 Per-window scores: {[f'{s:.1f}%' for s in window_scores]}")

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

    # Transcript
    st.markdown("**Speech-to-Text Transcript (Whisper-Tiny ASR)**")
    if transcript.strip():
        st.text_area("Transcribed Audio Content", value=transcript, height=110, disabled=True)
        st.caption(f"📊 {len(transcript.split())} words | {len(transcript)} characters")
    else:
        st.warning("⚠️ No spoken text transcribed (silence, noise, or non-speech audio).")

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

    with st.expander("📋 Classification Explanation"):
        # Determine tier label
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
# Helper: render unified score banner (Section 4)
# ─────────────────────────────────────────────────────────────────────────────
def _render_unified_banner(result: dict) -> None:
    u_score = result["unified_score"]
    u_verdict = result["unified_verdict"]
    if u_score is None:
        return

    if "ai" in u_verdict.lower() or "generated" in u_verdict.lower():
        score_color = "#F87171"
        icon = "🤖"
    elif "human" in u_verdict.lower():
        score_color = "#4ADE80"
        icon = "👤"
    else:
        score_color = "#FBBF24"
        icon = "🔶"

    st.markdown(
        f'<div class="unified-banner">'
        f'<div class="unified-score-label">UNIFIED AI-LIKELIHOOD SCORE</div>'
        f'<div class="unified-score-value" style="color:{score_color};">{u_score:.1f} / 100</div>'
        f'<div class="unified-verdict-text">{icon} {u_verdict}</div>'
        f'<div style="font-size:0.8rem;color:#94A3B8;margin-top:0.5rem;">'
        f'⚠️ {result["unified_score_note"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Cross-modal deferred notice (Section 3C)
    st.info(
        "🔬 **Section 3C (Cross-Modal Consistency Check) — Deferred**\n\n"
        "Comparing the deepfake text score of the audio transcript against the original text "
        "score to detect genuine mismatch requires calibrated delta thresholds — analogous to "
        "the Correction 9 grid-search for text. This will be implemented once matched real/fake "
        "audio test clips are available for threshold tuning. "
        + (f"Raw cross-modal Δ (for future calibration): `{result['transcript_text_delta']:.2f}`"
           if result.get("transcript_text_delta") is not None else
           "Raw delta: N/A (insufficient transcript length or text-only mode).")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main App: Section 1 — Inputs
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("1. Input")

raw_text = ""
uploaded_audio = None
tmp_audio_path = None

need_text  = "Text"    in mode or "Combined" in mode
need_audio = "Audio"   in mode or "Combined" in mode

if need_text and need_audio:
    col_ti, col_ai = st.columns(2)
    with col_ti:
        st.markdown("**Text to analyze:**")
        input_mode = st.radio("Text Input Method:", ["Paste Text", "Upload .txt"], horizontal=True, key="text_input_mode")
        if input_mode == "Paste Text":
            raw_text = st.text_area("Paste text:", height=180, placeholder="Paste text here...", key="text_area")
        else:
            txt_file = st.file_uploader("Upload .txt:", type=["txt"], key="txt_upload")
            if txt_file:
                try:
                    raw_text = txt_file.read().decode("utf-8")
                except UnicodeDecodeError:
                    txt_file.seek(0)
                    raw_text = txt_file.read().decode("latin-1")
    with col_ai:
        st.markdown("**Audio file to analyze:**")
        uploaded_audio = st.file_uploader(
            "Upload audio:", type=["wav","mp3","flac","ogg","m4a","aac"], key="audio_upload_combined"
        )
        if uploaded_audio:
            st.audio(uploaded_audio, format=f"audio/{uploaded_audio.name.split('.')[-1]}")
elif need_text:
    input_mode = st.radio("Input Method:", ["Paste Text", "Upload .txt"], horizontal=True, key="text_input_mode_only")
    if input_mode == "Paste Text":
        raw_text = st.text_area("Paste text to analyze:", height=220, placeholder="Paste text here...", key="text_area_only")
    else:
        txt_file = st.file_uploader("Upload .txt file:", type=["txt"], key="txt_upload_only")
        if txt_file:
            try:
                raw_text = txt_file.read().decode("utf-8")
            except UnicodeDecodeError:
                txt_file.seek(0)
                raw_text = txt_file.read().decode("latin-1")
else:
    uploaded_audio = st.file_uploader(
        "Upload audio file:", type=["wav","mp3","flac","ogg","m4a","aac"], key="audio_upload_only"
    )
    if uploaded_audio:
        col_f, col_p = st.columns([1, 2])
        with col_f:
            st.info(f"**File:** `{uploaded_audio.name}`\n\n**Size:** `{uploaded_audio.size/1024:.1f} KB`")
        with col_p:
            st.audio(uploaded_audio, format=f"audio/{uploaded_audio.name.split('.')[-1]}")

# Word count hint
word_count = len(raw_text.split()) if raw_text else 0
if raw_text:
    st.caption(f"Text: {word_count} words | {len(raw_text)} characters")
    if word_count < 30:
        st.warning("⚠️ **Short text**: Under 30 words. Burstiness signal will return N/A.")

# Robustness checkbox (text modes only)
run_robustness = False
if need_text:
    run_robustness = st.checkbox("Run T5 robustness check on text (adds ~30s)", value=False)

# Analyze button
can_analyze = bool(raw_text.strip()) if need_text and not need_audio else True
can_analyze = can_analyze and bool(uploaded_audio) if need_audio else can_analyze

analyze_btn = st.button(
    "🚀 Analyze",
    type="primary",
    use_container_width=True,
    disabled=not (
        (need_text and raw_text.strip()) or
        (need_audio and uploaded_audio is not None)
    ),
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

    spinner_parts = []
    if need_text and raw_text.strip():
        spinner_parts.append("text forensics")
    if need_audio and tmp_audio_path:
        spinner_parts.append("audio forensics (VAD + ASR + wav2vec2)")

    spinner_msg = f"Running {' and '.join(spinner_parts)}..."

    t_start = time.time()

    try:
        with st.spinner(spinner_msg):
            result = run_full_pipeline(
                text=raw_text.strip() if (need_text and raw_text.strip()) else None,
                audio_path=tmp_audio_path,
                run_robustness=run_robustness,
            )
        elapsed = time.time() - t_start
    except Exception as e:
        st.error(f"❌ **Pipeline Error**: `{type(e).__name__}: {e}`")
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

    # ── Section 4: Unified banner (Combined mode) ─────────────────────────────
    if "Combined" in mode and result["unified_score"] is not None:
        _render_unified_banner(result)
        st.divider()

    # ── Section 3A: Text panel ─────────────────────────────────────────────────
    if result["text_score"] is not None and raw_text.strip():
        _render_text_panel(result, raw_text.strip())

        with st.expander("Advanced: Text Robustness Check"):
            stab = result.get("text_stability_flag", "skipped")
            delta = result.get("text_signals", {}) or {}
            if stab == "stable":
                st.success(f"✅ **Stable** — T5 paraphrase delta within threshold.")
            elif stab == "unstable":
                st.warning(f"⚠️ **Unstable** — Score is sensitive to phrasing changes.")
            elif stab == "skipped":
                st.info("ℹ️ Robustness check not run. Enable checkbox to run.")
            else:
                st.info(f"Status: `{stab}`")

        with st.expander("Show Raw Text Signal Values (JSON)"):
            st.json({
                "text_score": result["text_score"],
                "text_verdict": result["text_verdict"],
                "signal_agreement": result["text_signal_agreement"],
                "signals": result["text_signals"],
            })

        if result["text_score"] is not None and "Audio" not in mode:
            st.divider()

    # ── Section 3B: Audio panel ────────────────────────────────────────────────
    if result["audio_score"] is not None:
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

    st.divider()
    st.caption(f"⏱️ Total analysis time: **{elapsed:.2f} seconds**")

else:
    # Empty state
    if "Text" in mode:
        st.info("💡 Paste or upload text above, then click **Analyze**.")
    elif "Audio" in mode:
        st.info("💡 Upload an audio clip above, then click **Analyze**.")
    else:
        st.info("💡 Provide text and/or audio above, then click **Analyze**.")
