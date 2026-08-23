import streamlit as st
from pages import drowsiness, emotion

st.set_page_config(page_title="NeuroDrive", layout="wide")
st.title("🧠 NeuroDrive: Smart Drowsiness & Emotion-Aware Driving Assistant")

tab1, tab2 = st.tabs(["Drowsiness Detection", "Emotion Analysis"])

with tab1:
    drowsiness.run()

with tab2:
    emotion.run()
