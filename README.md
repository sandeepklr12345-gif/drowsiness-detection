# 🧠 NeuroDrive AI Assistant

### Multimodal Driver Drowsiness, Emotion & Voice Fatigue Monitoring

NeuroDrive AI Assistant is a Python-based AI application that brings together **drowsiness monitoring, facial emotion analysis, and voice fatigue monitoring** in a unified Streamlit interface.

The project explores how computer vision, facial analysis, and audio signal processing can be used to study observable indicators associated with driver alertness.

---

## 📌 Overview

Driver fatigue can affect attention, reaction time, and decision-making. NeuroDrive provides an experimental platform for exploring visual and vocal cues that may be associated with reduced alertness.

The application is organized into three modules:

- 😴 **Drowsiness Monitoring:** Explores visual indicators associated with eye state and alertness.
- 😊 **Emotion AI:** Uses facial analysis to examine emotion-related patterns.
- 🎤 **Voice Fatigue Monitoring:** Explores audio characteristics using signal-processing techniques.

The modules are accessible through a Streamlit dashboard.

> **Note:** This is an educational and experimental project, not a certified driver-safety or medical system.

## ✨ Features

### 😴 Drowsiness Monitoring
- Computer-vision-based facial analysis.
- Exploration of visual cues associated with reduced alertness.
- Dedicated interface for the drowsiness module.

### 😊 Emotion AI
- Facial analysis using DeepFace and related tools.
- Exploration of emotion-related facial patterns.
- Dedicated interface for emotion analysis.

### 🎤 Voice Fatigue Monitoring
- Audio signal processing with librosa and SciPy.
- Audio input support through PyAudio, where configured.
- Exploration of acoustic features associated with voice characteristics.

### 🖥️ Interactive Dashboard
- Streamlit-based user interface.
- Navigation between the three monitoring modules.
- Modular structure for further experimentation and development.

## 🏗️ System Architecture

The application uses a modular architecture with three monitoring components.

```text
                    User
                      |
                      v
             Streamlit Application
                    (app.py)
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
     Drowsiness    Emotion      Voice Fatigue
     Monitoring   Recognition   Monitoring
          |           |           |
          v           v           v
      Analysis     Analysis     Analysis
          |           |           |
          +-----------+-----------+
                      |
                      v
              Display Results
               in Streamlit
```

**Note:** Each module performs its own analysis. This architecture does not imply a combined or validated fatigue prediction.



## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Application framework | Streamlit |
| Computer vision | OpenCV |
| Facial landmark processing | dlib |
| Facial analysis | DeepFace |
| Audio processing | librosa, PyAudio |
| Numerical computing | NumPy, SciPy |
| Machine learning | scikit-learn |
| Visualization | Matplotlib |

## 📁 Project Structure

```text
drowsiness-detection/
├── app.py
├── requirements.txt
├── filter_fer_folders.py
├── pages/
│   ├── Drowsiness.py
│   ├── Emotion.py
│   └── VoiceFatigue.py
├── models/
├── fer2013/
├── fer_selected/
├── sounds/
└── .gitignore
```

This is a representative overview of the repository. Individual directories may contain additional files and assets.

## 🚀 Getting Started

### Prerequisites

- Python compatible with the project's dependencies
- pip
- A webcam for visual modules that require camera input
- A microphone for audio modules that require live recording

Some dependencies, particularly dlib and PyAudio, may require operating-system-specific setup.

### 1. Clone the repository

```bash
git clone https://github.com/sandeepklr12345-gif/drowsiness-detection.git
cd drowsiness-detection
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

From the project root directory:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal. Streamlit commonly uses:

`http://localhost:8501`

### 5. Explore the modules

Navigate through the dashboard to access drowsiness monitoring, emotion analysis, and voice fatigue monitoring.

Camera, microphone, and model requirements depend on the selected module.

## 🧪 AI and Signal Processing Concepts

This project explores several areas of applied AI:

- **Computer vision:** Processing visual input and facial information.
- **Facial analysis:** Examining facial features and expression-related patterns.
- **Audio signal processing:** Extracting and analyzing characteristics of audio signals.
- **Machine learning:** Applying computational techniques to input features.
- **Multimodal monitoring:** Exploring complementary visual and vocal indicators.

The effectiveness of these approaches must be established through appropriate testing and evaluation rather than assumed from the use of AI libraries.

## 🔒 Privacy and Responsible Use

- Obtain permission before accessing camera or microphone input.
- Avoid storing personal media unnecessarily.
- Respect dataset licenses and usage restrictions.
- Treat emotion and fatigue outputs as uncertain model estimates.
- Do not use this prototype as the sole basis for driving-safety decisions.

## 🔮 Future Enhancements

- Integrate the modules into a validated multimodal assessment pipeline.
- Improve robustness under different lighting and recording conditions.
- Add confidence scores and explainable outputs.
- Evaluate model performance using precision, recall, F1-score, and confusion matrices.
- Add historical monitoring and result visualization.
- Improve automated testing and error handling.
- Explore edge-device deployment.

These are potential enhancements, not claims about existing functionality.

## 🎓 Learning Outcomes

This project provides practical experience with:

- Python application development.
- Computer vision and facial analysis.
- Audio signal processing.
- Streamlit dashboard development.
- Machine-learning library integration.
- Modular application design.
- Responsible handling of human-related data.

## 👨‍💻 Author

**Sandeep K**

GitHub: [@sandeepklr12345-gif](https://github.com/sandeepklr12345-gif)

Repository: [drowsiness-detection](https://github.com/sandeepklr12345-gif/drowsiness-detection)

---

*Exploring computer vision and audio intelligence for driver-state monitoring.*
