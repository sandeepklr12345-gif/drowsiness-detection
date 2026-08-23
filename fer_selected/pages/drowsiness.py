import streamlit as st
import cv2
import dlib
import numpy as np
from scipy.spatial import distance

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def run():
    st.header("👀 Drowsiness Detection")
    st.write("Live drowsiness detection using eye aspect ratio (EAR)")

    EAR_THRESHOLD = 0.25
    EAR_CONSEC_FRAMES = 20
    COUNTER = 0

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)

    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = np.array([[p.x, p.y] for p in shape.parts()])

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < EAR_THRESHOLD:
                    COUNTER += 1
                    if COUNTER >= EAR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    COUNTER = 0

                for (x, y) in np.concatenate((leftEye, rightEye)):
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
    except Exception as e:
        st.error("Error: " + str(e))
    finally:
        cap.release()
