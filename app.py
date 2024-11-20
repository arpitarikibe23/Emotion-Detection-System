import streamlit as st
from PIL import Image
import numpy as np
from fer import FER
from dotenv import load_dotenv
from deepface import DeepFace
import cv2
import tensorflow as tf

# Load environment variables
load_dotenv()

# Display TensorFlow version for debugging
st.write(f"TensorFlow Version: {tf.__version__}")

# Function to detect emotions using DeepFace and FER as fallback
def detect_emotions(image):
    if image.mode != "RGB":
        image = image.convert("RGB")  # Convert image to RGB if necessary

    image_np = np.array(image)

    try:
        # Analyze emotions using DeepFace
        analysis = DeepFace.analyze(image_np, actions=["emotion"], enforce_detection=False)
        if analysis:
            return analysis[0]["emotions"]
    except Exception as e:
        st.error(f"DeepFace Error: {e}")

    # Fallback to FER
    detector = FER(mtcnn=True)
    emotions = detector.detect_emotions(image_np)
    if emotions:
        return emotions[0]["emotions"]

    return None

# Function to generate analysis and advice based on emotions
def analyze_emotions_with_advice(emotions):
    dominant_emotion = max(emotions, key=emotions.get)
    advice = {
        "happy": "You seem happy! Keep spreading positivity.",
        "sad": "Feeling sad? Try connecting with a friend or activity.",
        "neutral": "Feeling neutral? Take a moment to reflect.",
        "angry": "Feeling angry? Deep breaths can help.",
        "surprise": "Surprised? Embrace the unexpected!",
        "fear": "Feeling fear? Ensure safety and talk to someone you trust."
    }.get(dominant_emotion, "Stay positive and take care!")

    emotion_summary = ", ".join([f"{k}: {v:.2f}" for k, v in emotions.items()])
    return f"Emotions detected: {emotion_summary}\n\nAdvice: {advice}"

# Streamlit Interface
st.title("Emotion Detection System")
tab1, tab2 = st.tabs(["Image Analysis", "Live Video Analysis"])

with tab1:
    st.header("Image Analysis")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if st.button("Analyze Image"):
        if uploaded_file:
            image = Image.open(uploaded_file)
            emotions = detect_emotions(image)
            if emotions:
                result = analyze_emotions_with_advice(emotions)
                st.write(result)
            else:
                st.error("No emotions detected.")
        else:
            st.warning("Please upload an image.")

with tab2:
    st.header("Live Video Analysis")
    if st.button("Start Video Analysis"):
        video_capture = cv2.VideoCapture(0)
        if video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Captured Frame", use_column_width=True)
                image = Image.fromarray(frame_rgb)
                emotions = detect_emotions(image)
                if emotions:
                    result = analyze_emotions_with_advice(emotions)
                    st.write(result)
                else:
                    st.warning("No emotions detected.")
            video_capture.release()
        else:
            st.error("Failed to access webcam.")
