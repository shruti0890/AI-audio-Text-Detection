"""
Audio Forensics Streamlit POC Application
==========================================
Interactive Web Interface for Audio Deepfake Classification & Speech-to-Text Analysis.

Features:
- Multi-format audio file upload (.wav, .mp3, .flac, .ogg, .m4a, .aac)
- Integrated Audio Player
- Automated Speech-to-Text Transcription via Whisper-Tiny
- Neural Deepfake Classification via wav2vec2-deepfake-voice-detector
- Complete Logit Margins & Sigmoid Scoring Formula breakdown
- Human vs AI Verdict Badge
- Plain-Language Analytical Reason & Confidence Explanation

Run instructions:
    streamlit run audio_forensics/app.py
"""

import os
import tempfile
import streamlit as st
import numpy as np
import torch

from audio_forensics.pipeline import analyze_audio

# Page config & Custom Styling
st.set_page_config(
    page_title="Audio Forensics & Deepfake Detection POC",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .verdict-box-human {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .verdict-box-fake {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .verdict-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .verdict-human-title { color: #047857; }
    .verdict-fake-title { color: #B91C1C; }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
    }
    .reason-card {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Info ─────────────────────────────────────────────────────────────
st.sidebar.title("🎙️ Audio Forensics Engine")
st.sidebar.markdown("""
**Pipeline Architecture:**
1. **VAD**: Silero VAD (Silence Stripping & 16kHz Mono Resampling)
2. **ASR**: OpenAI Whisper-Tiny (`automatic-speech-recognition`)
3. **Deepfake Detector**: `garystafford/wav2vec2-deepfake-voice-detector`

---
**Mathematical Formula:**
$$S_{Audio} = \\text{Sigmoid}(\\text{logit}_{fake} - \\text{logit}_{real}) \\times 100$$

- **Score → 100%**: High AI Synthetic Confidence
- **Score → 0%**: High Real Human Confidence
""")

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎙️ Audio Forensics & Deepfake Detection POC</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload an audio clip to extract spoken transcript, compute raw neural logit margins, and classify audio as Human or AI Voice.</div>', unsafe_allow_html=True)

# ── File Upload Section ──────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a", "aac", "wma"],
    help="Supports WAV, MP3, FLAC, OGG, M4A, AAC audio formats."
)

if uploaded_file is not None:
    # Display file details & Audio Player
    col_file, col_player = st.columns([1, 2])
    with col_file:
        st.info(f"**Filename:** `{uploaded_file.name}`\n\n**Size:** `{uploaded_file.size / 1024:.1f} KB`")
    with col_player:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

    if st.button("🚀 Analyze Audio Clip", type="primary", use_container_width=True):
        with st.spinner("Running VAD silence stripping, Whisper-Tiny ASR, and wav2vec2 scoring..."):
            # Save uploaded bytes to temporary file for librosa ingestion
            file_ext = os.path.splitext(uploaded_file.name)[1]
            if not file_ext:
                file_ext = ".wav"
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Execute audio forensics pipeline
                results = analyze_audio(tmp_path)
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        st.divider()

        # Extract metrics
        audio_score = results["audio_score"]
        logit_fake = results["logit_fake"]
        logit_real = results["logit_real"]
        transcript = results["transcript"]
        logit_diff = logit_fake - logit_real
        is_ai = audio_score >= 50.0

        # ── 1. Final Verdict Display ──────────────────────────────────────────
        st.subheader("1. Classification Verdict")
        if is_ai:
            st.markdown(f"""
            <div class="verdict-box-fake">
                <div class="verdict-title verdict-fake-title">🤖 AI-GENERATED / DEEPFAKE VOICE</div>
                <div style="font-size: 1.1rem; color: #991B1B;">Synthetic Risk Score: <b>{audio_score:.2f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-box-human">
                <div class="verdict-title verdict-human-title">👤 REAL HUMAN VOICE</div>
                <div style="font-size: 1.1rem; color: #065F46;">Authentic Voice Confidence: <b>{100.0 - audio_score:.2f}%</b> (Synthetic Score: {audio_score:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        # ── 2. Speech-to-Text Transcript Panel ────────────────────────────────
        st.subheader("2. Speech-to-Text Transcript (Whisper-Tiny ASR)")
        if transcript.strip():
            st.text_area("Transcribed Spoken Content", value=transcript, height=100, disabled=True)
            st.caption(f"📊 Stats: **{len(transcript.split())}** words | **{len(transcript)}** characters")
        else:
            st.warning("⚠️ No clear spoken text was transcribed from the audio clip (tone/silence/noise).")

        # ── 3. Scoring Margins Breakdown ──────────────────────────────────────
        st.subheader("3. Neural Scoring Margins & Logits Breakdown")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.metric("Final Deepfake Score (S_Audio)", f"{audio_score:.2f}%")
        with col_m2:
            st.metric("Raw Logit (Fake)", f"{logit_fake:.4f}")
        with col_m3:
            st.metric("Raw Logit (Real)", f"{logit_real:.4f}")
        with col_m4:
            st.metric("Logit Difference Δ", f"{logit_diff:+.4f}")

        # Progress bar representation
        st.markdown("**Synthetic Risk Bar:**")
        st.progress(min(max(audio_score / 100.0, 0.0), 1.0))

        # Formula mathematical explanation box
        st.markdown(f"""
        > **Scoring Equation Executed:**
        > $$S_{{Audio}} = \\text{{Sigmoid}}({logit_fake:.4f} - ({logit_real:.4f})) \\times 100 = \\text{{Sigmoid}}({logit_diff:+.4f}) \\times 100 = {audio_score:.2f}\\%$$
        """)

        # ── 4. Detailed Reason & Analytical Explanation ───────────────────────
        st.subheader("4. Classification Reason & Explanation")

        # Determine confidence level
        abs_diff = abs(logit_diff)
        if abs_diff > 3.0:
            confidence_str = "Extreme Confidence"
        elif abs_diff > 1.5:
            confidence_str = "High Confidence"
        elif abs_diff > 0.5:
            confidence_str = "Moderate Confidence"
        else:
            confidence_str = "Borderline / Low Confidence"

        if is_ai:
            reason_text = (
                f"**Why classified as AI-Generated?**\n\n"
                f"- **Logit Dominance:** The model's raw pre-softmax logit for synthetic voice (`logit_fake` = `{logit_fake:.4f}`) "
                f"exceeds the logit for authentic voice (`logit_real` = `{logit_real:.4f}`) by a margin of **`{logit_diff:+.4f}`**.\n"
                f"- **Sigmoid Mapping:** Passing this positive margin into the Sigmoid activation yields a final risk score of **`{audio_score:.2f}%`** (above the 50.0% deepfake decision threshold).\n"
                f"- **Confidence Assessment:** **{confidence_str}** (Margin Δ = `{abs_diff:.4f}`).\n"
                f"- **Acoustic Characteristics:** The wav2vec2 transformer layer detected spectral artifacts, pitch irregularities, or phase mismatches typical of synthetic speech synthesis engines."
            )
        else:
            reason_text = (
                f"**Why classified as Real Human Voice?**\n\n"
                f"- **Logit Dominance:** The model's raw pre-softmax logit for authentic human voice (`logit_real` = `{logit_real:.4f}`) "
                f"is greater than or equal to the logit for synthetic voice (`logit_fake` = `{logit_fake:.4f}`) with a margin of **`{logit_diff:+.4f}`**.\n"
                f"- **Sigmoid Mapping:** The resulting logit difference produces a deepfake risk score of **`{audio_score:.2f}%`** (below the 50.0% threshold), confirming human voice characteristics with **`{100.0 - audio_score:.2f}%`** confidence.\n"
                f"- **Confidence Assessment:** **{confidence_str}** (Margin Δ = `{abs_diff:.4f}`).\n"
                f"- **Acoustic Characteristics:** Acoustic feature representations align with natural vocal tract resonances and natural human speech dynamics."
            )

        st.markdown(f'<div class="reason-card">{reason_text}</div>', unsafe_allow_html=True)

        st.caption("ℹ️ *Note: `wer_confidence_note` internal flag:* `" + results.get("wer_confidence_note", "") + "`")

else:
    st.info("💡 Upload an audio clip above to start the analysis.")
