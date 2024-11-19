import streamlit as st
import os
from PIL import Image
import numpy as np
from fer import FER
import cv2
from dotenv import load_dotenv
import tensorflow as tf

# Print TensorFlow version for debugging
print(tf.__version__)

# Load API key from .env file
load_dotenv()

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

    # Initialize FER detector
    detector = FER(mtcnn=True)

    # Detect emotions in the image
    emotions = detector.detect_emotions(image_np)

    if emotions:
        return emotions[0]['emotions']
    return None

# Function to analyze detected emotions with a summary and advice
def analyze_emotions_with_advice(emotions):
    emotion_analysis = ", ".join([f"{emotion}: {score:.2f}" for emotion, score in emotions.items()])
    summary = f"The detected emotions are: {emotion_analysis}."

    # Generate advice based on emotions
    dominant_emotion = max(emotions, key=emotions.get)
    advice = ""
    if dominant_emotion == "sad":
        advice = "You seem to be feeling sad. Consider speaking to a friend or engaging in an activity you enjoy."
    elif dominant_emotion == "happy":
        advice = "You seem happy! Keep spreading positivity and cherish this moment."
    elif dominant_emotion == "neutral":
        advice = "You appear neutral. If you feel like it, take a moment to reflect on your thoughts."
    elif dominant_emotion in ["angry", "disgust"]:
        advice = "It seems like you might be feeling angry or frustrated. Try to relax, take deep breaths, and focus on calming activities."
    elif dominant_emotion == "fear":
        advice = "You may be feeling fear. Take a moment to ensure your safety and consider sharing your feelings with someone you trust."
    elif dominant_emotion == "surprise":
        advice = "You seem surprised! Embrace the unexpected and enjoy the moment."

    return summary + "\n\nAdvice: " + advice

# Streamlit App
st.title("AI-Powered Depression and Emotion Detection System")
st.text("Use the AI system for detecting depression and emotions from images and live video.")

# Tabs for different functionalities
tab1, tab2 = st.tabs(["Image Analysis", "Live Video Analysis"])

# Image Analysis Tab
with tab1:
    st.header("Image Analysis")
    uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
    submit_image = st.button('Analyze Image')

    if submit_image:
        if uploaded_file:
            image = Image.open(uploaded_file)
            emotions = detect_emotions(image)
            if emotions:
                response = analyze_emotions_with_advice(emotions)
                st.write(response)
            else:
                st.write("No emotions detected in the image.")
        else:
            st.warning("Please upload an image before submitting.")

# Live Video Analysis Tab
with tab2:
    st.header("Live Video Analysis")
    capture_frame = st.button('Start Live Video and Analyze Frame')

    if capture_frame:
        # Initialize the webcam
        video_capture = cv2.VideoCapture(0)  # 0 for the default camera
        if not video_capture.isOpened():
            st.error("Failed to access the webcam. Ensure you have allowed camera access in your browser.")
        else:
            with st.spinner("Accessing webcam..."):
                # Read a single frame from the webcam
                ret, frame = video_capture.read()
                if ret:
                    # Convert frame to RGB format
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Display the frame
                    st.image(frame_rgb, caption="Live Video Frame", use_container_width=True)

                    # Convert frame to a PIL Image for processing
                    image = Image.fromarray(frame_rgb)

                    # Detect emotions
                    emotions = detect_emotions(image)
                    if emotions:
                        response = analyze_emotions_with_advice(emotions)
                        st.write(response)
                    else:
                        st.warning("No emotions detected in the frame.")
                else:
                    st.error("Failed to capture video frame. Check if another app is using the camera.")

            # Release the webcam after processing
            video_capture.release()
            cv2.destroyAllWindows()

