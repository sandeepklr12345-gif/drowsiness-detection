import cv2
import dlib
import numpy as np
import joblib
import streamlit as st
import time
from pathlib import Path
from collections import deque

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "models"

def run():

    st.markdown("<h2 style='text-align:center;'>🧠 Emotion Monitoring</h2>", unsafe_allow_html=True)
    st.write("System detects **Anger**, **Fatigue**, and **Stress** in real-time.")

    # --- Model Paths ---
    model_path = MODELS_DIR / "emotion_model.pkl"
    shape_path = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"

    if not model_path.exists():
        st.error("❌ Emotion model missing!")
        return
    if not shape_path.exists():
        st.error("❌ Face landmark model missing!")
        return

    # --- Load Model ---
    model = joblib.load(model_path)
    labels = ['Anger', 'Fatigue', 'Stress']
    responses = {
        "Anger": "😠 Calm down — take deep breaths.",
        "Fatigue": "😴 You look tired — take a short break.",
        "Stress": "😓 High stress — relax & breathe slowly.",
        "Normal": "😊 You look relaxed!"
    }

    # --- Dlib Models ---
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(shape_path))

    # --- UI Buttons ---
    if "emotion_active" not in st.session_state:
        st.session_state.emotion_active = False

    col1, col2 = st.columns(2)
    if col1.button("▶ Start Emotion Detection"):
        st.session_state.emotion_active = True
    if col2.button("⏹ Stop"):
        st.session_state.emotion_active = False

    # --- UI Display Slots ---
    status_box = st.empty()
    output_text = st.empty()
    frame_window = st.image([])

    # --- Emotion Buffer (3-sec stability) ---
    buffer = deque(maxlen=30)
    last_emotion = "Normal"
    last_time = time.time()

    # --- Camera Loop ---
    if st.session_state.emotion_active:
        cap = cv2.VideoCapture(0)

        while st.session_state.emotion_active:

            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Camera not available")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            current = "Normal"

            for f in faces:
                x1, y1, x2, y2 = f.left(), f.top(), f.right(), f.bottom()
                roi = frame[y1:y2, x1:x2]

                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_resized = cv2.resize(roi_gray, (64, 64)) / 255.0
                roi_flat = roi_resized.flatten().reshape(1, -1)

                pred = model.predict(roi_flat)[0]
                current = labels[pred] if pred < len(labels) else "Normal"

            buffer.append(current)

            # Show emotion only if stable for 3 seconds
            if time.time() - last_time > 3:
                last_emotion = max(set(buffer), key=buffer.count)
                last_time = time.time()

            emotion = last_emotion
            color = "#00c853" if emotion == "Normal" else "#ff1744"

            # ✅ GLASS EMOTION CARD
            glow = "animation:pulse 2s infinite;" if emotion!="Normal" else ""
            status_box.markdown(
                f"""
                <style>
                @keyframes pulse {{
                    0% {{ box-shadow:0 0 8px {color}55, 0 0 15px {color}55; }}
                    50% {{ box-shadow:0 0 25px {color}, 0 0 45px {color}; }}
                    100% {{ box-shadow:0 0 8px {color}55, 0 0 15px {color}55; }}
                }}

                .glass-box {{
                    backdrop-filter: blur(12px);
                    -webkit-backdrop-filter: blur(12px);
                    background: rgba(255,255,255,0.08);
                    border: 1px solid rgba(255,255,255,0.25);
                    padding: 14px;
                    border-radius: 18px;
                    font-size: 22px;
                    font-weight: 700;
                    color: {color};
                    width: 60%;
                    margin: 12px auto;
                    text-align:center;
                    transition: 0.3s;
                }}

                .pulse {{
                    border: 2px solid {color};
                    {glow}
                }}
                </style>

                <div class="glass-box {'pulse' if emotion!='Normal' else ''}">
                    Emotion: {emotion}
                </div>
                """,
                unsafe_allow_html=True
            )

            output_text.markdown(f"### {responses[emotion]}")

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            time.sleep(0.05)

        cap.release()
