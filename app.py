
import streamlit as st
import google.generativeai as genai
import os
from PIL import Image
import numpy as np
from fer import FER
import cv2
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Configure the PaLM API with the correct API key
api_key = os.getenv("AIzaSyAyL-cstbn9eoY-90XcVnGCB4Qnug2ztVA")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("API Key not found. Please set it in the .env file.")

# Function to analyze image for depression and emotion detection using FER
def detect_emotions(image):
    # Ensure the image has 3 channels (convert RGBA to RGB if necessary)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    elif image.mode != "RGB":
        st.warning("Uploaded image has an unsupported mode. Converting to RGB.")
        image = image.convert("RGB")

    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Use FER to detect emotions
    detector = FER(mtcnn=True)
    emotions = detector.detect_emotions(image_np)

    if emotions:
        return emotions[0]['emotions']
    return None

# Function to analyze detected emotions with a summary
def analyze_emotions(emotions):
    emotion_analysis = ", ".join([f"{emotion}: {score:.2f}" for emotion, score in emotions.items()])
    summary = f"The detected emotions are: {emotion_analysis}. Based on these emotions, consider seeking expert advice if needed."
    return summary

# Streamlit App
st.title("AI-Powered Depression and Emotion Detection System")
st.text("Use the AI system for detecting depression and emotions from images and live video.")

# Tabs for different functionalities
tab1, tab2 = st.tabs(["Image Analysis", "Live Video Analysis"])

with tab1:
    st.header("Image Analysis")
    uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
    submit_image = st.button('Analyze Image')

    if submit_image:
        if uploaded_file:
            image = Image.open(uploaded_file)
            emotions = detect_emotions(image)
            if emotions:
                response = analyze_emotions(emotions)
                st.write(response)
            else:
                st.write("No emotions detected in the image.")

with tab2:
    st.header("Live Video Analysis")
    capture_frame = st.button('Capture and Analyze Frame')
    if capture_frame:
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            emotions = detect_emotions(image)
            if emotions:
                response = analyze_emotions(emotions)
                st.write(response)
            else:
                st.write("No emotions detected.")
        else:
            st.write("Failed to capture video frame.")

