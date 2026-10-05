"""Streamlit dashboard. Run with:  streamlit run app.py"""
import hashlib
import io

import streamlit as st

import audio_processing
from predict import SpeechEmotionPipeline

st.set_page_config(page_title="Speech Emotion Analysis", page_icon="🎙️", layout="wide")

COLORS = {"happy": "#F59E0B", "sad": "#3B82F6", "angry": "#EF4444",
          "neutral": "#6B7280", "fear": "#8B5CF6", "surprise": "#EC4899"}
EMOJI = {"happy": "😊", "sad": "😢", "angry": "😠",
         "neutral": "😐", "fear": "😨", "surprise": "😲"}

CSS = """
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
footer {visibility: hidden;}
.hero {background: linear-gradient(120deg, #4f46e5, #7c3aed 55%, #db2777);
       padding: 28px 32px; border-radius: 18px; margin-bottom: 22px;}
.hero-title {color: #ffffff; font-size: 2rem; font-weight: 800; line-height: 1.2;}
.hero-sub {color: rgba(255,255,255,.9); margin-top: 8px; font-size: 1.02rem;}
.pipeline {margin-top: 16px;}
.pipeline span {display: inline-block; color: #fff; background: rgba(255,255,255,.18);
                padding: 5px 14px; border-radius: 999px; margin: 0 8px 6px 0; font-size: .85rem;}
.pipeline span.arrow {background: none; padding: 5px 0;}
.stat {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 16px 20px;}
.stat-label {font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; opacity: .65;}
.stat-value {font-size: 1.7rem; font-weight: 700; margin-top: 6px;}
.badge {display: inline-block; color: #fff; padding: 4px 16px; border-radius: 999px; font-size: 1.3rem;}
.bar-row {display: flex; align-items: center; gap: 12px; margin: 12px 0;}
.bar-label {width: 115px; font-weight: 600;}
.bar-track {flex: 1; height: 12px; border-radius: 999px; background: rgba(128,128,128,.2); overflow: hidden;}
.bar-fill {height: 100%; border-radius: 999px;}
.bar-value {width: 44px; text-align: right;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models (the first run downloads them, please wait)...")
def get_pipeline():
    return SpeechEmotionPipeline()


def stat_card(label, value_html):
    return (f'<div class="stat"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value_html}</div></div>')


def probability_bars(scores):
    rows = ""
    for name, p in sorted(scores.items(), key=lambda item: -item[1]):
        rows += (f'<div class="bar-row"><div class="bar-label">{EMOJI[name]} {name.capitalize()}</div>'
                 f'<div class="bar-track"><div class="bar-fill" '
                 f'style="width:{p * 100:.1f}%; background:{COLORS[name]};"></div></div>'
                 f'<div class="bar-value">{p:.0%}</div></div>')
    return rows


st.markdown(
    '<div class="hero"><div class="hero-title">🎙️ Real-Time Speech Emotion Analysis</div>'
    '<div class="hero-sub">Speak into the microphone and get the transcript, the emotion in your '
    'voice and an AI explanation within seconds.</div>'
    '<div class="pipeline"><span>🎤 Microphone</span><span class="arrow">→</span>'
    '<span>📝 Whisper</span><span class="arrow">→</span>'
    '<span>🎭 Wav2Vec2</span><span class="arrow">→</span><span>🤖 LLM</span></div></div>',
    unsafe_allow_html=True,
)

pipeline = get_pipeline()
st.session_state.setdefault("recorder_id", 0)

with st.container(border=True):
    st.markdown("#### 🎙️ Record your voice")
    st.caption("Speak for 3 to 15 seconds in any language, then click stop.")
    recording = st.audio_input(
        "Microphone", key=f"recorder_{st.session_state['recorder_id']}", label_visibility="collapsed"
    )
    if recording is not None and st.button("🔄 Record again"):
        st.session_state["recorder_id"] += 1
        st.session_state["recording_key"] = None
        st.session_state["result"] = None
        st.session_state["error"] = None
        st.rerun()

if recording is None:
    st.info("Click the microphone above and start speaking. Your results will appear here.")
    st.stop()

data = recording.getvalue()
key = hashlib.md5(data).hexdigest()

if st.session_state.get("recording_key") != key:      # analyse each new recording once
    st.session_state["recording_key"] = key
    st.session_state["result"], st.session_state["error"] = None, None
    try:
        audio = audio_processing.load_audio(io.BytesIO(data))
        with st.spinner("Analysing your voice..."):
            st.session_state["result"] = pipeline.run(audio)
    except ValueError as err:
        st.session_state["error"] = str(err)

if st.session_state.get("error"):
    st.warning(st.session_state["error"])

result = st.session_state.get("result")
if result:
    emotion = result["emotion"]
    timings = result["timings"]

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        stat_card("Detected emotion",
                  f'<span class="badge" style="background:{COLORS[emotion]};">'
                  f'{EMOJI[emotion]} {emotion.capitalize()}</span>'),
        unsafe_allow_html=True,
    )
    c2.markdown(stat_card("Confidence", f'{result["confidence"]:.0%}'), unsafe_allow_html=True)
    c3.markdown(stat_card("Processing time", f'{timings["total"]:.1f} s'), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1.1, 1])
    with left:
        with st.container(border=True):
            st.markdown("#### 📝 Transcript")
            st.write(result["text"] or "(no speech recognised)")
        with st.container(border=True):
            st.markdown("#### 🤖 AI analysis")
            st.write(result["analysis"])
    with right:
        with st.container(border=True):
            st.markdown("#### 🎭 Emotion probabilities")
            st.markdown(probability_bars(result["scores"]), unsafe_allow_html=True)

    st.caption("⏱️ " + "  |  ".join(f"{n} {s:.1f}s" for n, s in timings.items() if n != "total"))