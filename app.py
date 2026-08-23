import streamlit as st


st.set_page_config(page_title="NeuroDrive", layout="wide")

# ---- UI STYLE ----
st.markdown("""
<style>
.card {
    padding: 25px;
    background: linear-gradient(135deg,#1e1f25,#2d3347);
    border-radius: 15px;
    border: 1px solid #3a3f55;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
    text-align: center;
    transition: 0.2s ease-in-out;
}
.card:hover {
    transform: scale(1.04);
    border-color: #6f9bff;
}
.card h2 {
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
}
.card p {
    font-size: 14px;
    color: #bfc6d1;
}
button { border: none; background: none; padding: 0; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🧠 NeuroDrive AI Assistant</h1>", unsafe_allow_html=True)
st.write("### Smart Drowsiness, Emotion & Voice Fatigue Monitoring")

# ---- CARDS UI ----
col1, col2, col3 = st.columns(3)

with col1:
    if st.button(" ", key="drowsiness_btn"):
        st.switch_page("pages/Drowsiness.py")
    st.markdown("""
    <div class="card">
        <h2>😴 Drowsiness</h2>
        <p>Eye & Blink Monitoring</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if st.button(" ", key="emotion_btn"):
        st.switch_page("pages/Emotion.py")
    st.markdown("""
    <div class="card">
        <h2>😊 Emotion AI</h2>
        <p>Stress & Fatigue Tracking</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    if st.button(" ", key="voice_btn"):
        st.switch_page("pages/VoiceFatigue.py")
    st.markdown("""
    <div class="card">
        <h2>🎤 Voice Fatigue</h2>
        <p>Speech Stress Detection</p>
    </div>
    """, unsafe_allow_html=True)
