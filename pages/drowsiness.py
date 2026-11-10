import streamlit as st
import cv2
import dlib
import numpy as np
import threading
import simpleaudio as sa
import time
from pathlib import Path
from scipy.spatial import distance

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "models"
alert_sound_path = str(BASE_DIR / "sounds" / "alert.wav")

# --- Alert Sound ---
def play_alert_sound():
    try:
        wave_obj = sa.WaveObject.from_wave_file(alert_sound_path)
        wave_obj.play()
    except Exception as e:
        print(f"Sound error: {e}")

# --- EAR Calculation ---
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# ======================= UI + LOGIC =======================

st.markdown("<h1 style='text-align:center;'>😴 Drowsiness Detection</h1>", unsafe_allow_html=True)

st.info("Real-time blink & eye-closure monitoring using EAR (Eye Aspect Ratio)")

# UI Styling
st.markdown("""
<style>
.status-box {
    padding:10px;
    border-radius:8px;
    font-weight:bold;
    text-align:center;
    margin-bottom:10px;
}
.cam-box {
    border:2px solid #4c4f58;
    padding:10px;
    margin-top:10px;
    border-radius:10px;
}
.start-btn button {
    background-color:#00c853 !important;
    color:white !important;
    font-weight:bold;
    width:100%;
}
.stop-btn button {
    background-color:#d50000 !important;
    color:white !important;
    font-weight:bold;
    width:100%;
}
</style>
""", unsafe_allow_html=True)

if "run_drowsy" not in st.session_state:
    st.session_state.run_drowsy = False

# Buttons Row
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start", key="start", type="primary"):
        st.session_state.run_drowsy = True

with col2:
    if st.button("⏹ Stop", key="stop", type="primary"):
        st.session_state.run_drowsy = False

# Status box
status = "🟢 ACTIVE" if st.session_state.run_drowsy else "🔴 STOPPED"
color = "#00c853" if st.session_state.run_drowsy else "#d50000"

st.markdown(f"<div class='status-box' style='background:{color}22;color:{color};'>{status}</div>", unsafe_allow_html=True)

EAR_display = st.empty()
FRAME_WINDOW = st.image([])

# Drowsiness Logic
if st.session_state.run_drowsy:

    EAR_THRESHOLD = 0.25
    EAR_CONSEC_FRAMES = 20
    counter = 0
    alert = False

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        str(MODELS_DIR / "shape_predictor_68_face_landmarks.dat")
    )

    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)

    cap = cv2.VideoCapture(0)

    while st.session_state.run_drowsy:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Camera not detected!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            L = shape[lStart:lEnd]
            R = shape[rStart:rEnd]

            ear = (eye_aspect_ratio(L) + eye_aspect_ratio(R)) / 2.0
            EAR_display.markdown(f"### 👁 EAR: **{ear:.3f}**")

            if ear < EAR_THRESHOLD:
                counter += 1
                if counter >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "⚠ DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if not alert:
                        alert = True
                        threading.Thread(target=play_alert_sound, daemon=True).start()
            else:
                counter = 0
                alert = False

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        time.sleep(0.03)

    cap.release()
