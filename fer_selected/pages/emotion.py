import cv2
import dlib
import numpy as np
import joblib
import streamlit as st
import time

# Load the emotion model (Anger, Fatigue, Stress)
emotion_model = joblib.load("emotion_model.pkl")

# Emotion labels (ensure these match your model's training order)
emotion_labels = ['Anger', 'Fatigue', 'Stress']

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def run():
    st.header("😠 Real-Time Emotion Detection (Anger, Fatigue, Stress)")
    st.write("This app uses your webcam to detect emotion in real time.")

    run_button = st.button("Start Camera")

    if run_button:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access the webcam.")
            return

        FRAME_WINDOW = st.image([])  # For webcam display
        placeholder = st.empty()     # For emotion text

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Could not read frame from webcam.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                for face in faces:
                    landmarks = predictor(gray, face)

                    # Crop and preprocess face
                    face_img = frame[face.top():face.bottom(), face.left():face.right()]
                    if face_img.size == 0:
                        continue  # Skip empty detections

                    face_img = cv2.resize(face_img, (64, 64))
                    face_img = face_img.astype("float32") / 255.0
                    face_img = np.expand_dims(face_img, axis=0)

                    # Convert face image to feature vector (flatten it)
                    face_img = face_img.flatten()

                    # Predict emotion
                    prediction = emotion_model.predict([face_img])
                    emotion_idx = prediction[0]
                    emotion = emotion_labels[emotion_idx]

                    # Display detected emotion
                    placeholder.success(f"🧠 Emotion Detected: **{emotion.upper()}**")

                # Display the webcam feed
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(rgb_frame)

                # Add a small delay for better performance
                time.sleep(0.1)

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            cap.release()  # Release webcam
