
import streamlit as st
import cv2
from transformers import pipeline
from PIL import Image
import numpy as np
import streamlit as st
import cv2
from transformers import pipeline
from PIL import Image
import numpy as np
st.title("Emotion Detection System")
st.write("Upload an image or video to analyze emotions using Generative AI and LLMs.")
upload_type = st.radio("Choose input type:", ("Image", "Video"))
emotion_detector = pipeline("sentiment-analysis", model="mrm8488/t5-base-finetuned-emotion")
def analyze_emotion_image(image):
    image = Image.fromarray(image)
    emotion = emotion_detector(image)[0]  # Get prediction from the model
    return emotion['label'], emotion['score']
def analyze_emotion_video(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        label, score = analyze_emotion_image(frame)
        emotions.append(label)
        
        st.image(frame, channels="BGR")
        st.write(f"Detected Emotion: {label} (Score: {score:.2f})")

    cap.release()
if upload_type == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        label, score = analyze_emotion_image(np.array(image))
        st.write(f"Detected Emotion: {label} (Score: {score:.2f})")
elif upload_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        analyze_emotion_video("temp_video.mp4")

#streamlit run app.py
import streamlit as st
import cv2
from transformers import pipeline
from PIL import Image
import numpy as np

st.title("Emotion Detection System")
st.write("Upload an image or video to analyze emotions using Generative AI and LLMs.")
upload_type = st.radio("Choose input type:", ("Image", "Video"))

emotion_detector = pipeline("sentiment-analysis", model="mrm8488/t5-base-finetuned-emotion")

def analyze_emotion_image(image):
    image = Image.fromarray(image)
    emotion = emotion_detector(image)[0]
    return emotion['label'], emotion['score']

def analyze_emotion_video(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        label, score = analyze_emotion_image(frame)
        emotions.append(label)
        
        st.image(frame, channels="BGR")
        st.write(f"Detected Emotion: {label} (Score: {score:.2f})")

    cap.release()

if upload_type == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        label, score = analyze_emotion_image(np.array(image))
        st.write(f"Detected Emotion: {label} (Score: {score:.2f})")

elif upload_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        analyze_emotion_video("temp_video.mp4")
import streamlit as st
import cv2
from transformers import pipeline
from PIL import Image
import numpy as np

st.title("Emotion Detection System")
st.write("Upload an image or video to analyze emotions using Generative AI and LLMs.")
upload_type = st.radio("Choose input type:", ("Image", "Video"))

emotion_detector = pipeline("sentiment-analysis", model="mrm8488/t5-base-finetuned-emotion")

def analyze_emotion_image(image):
    image = Image.fromarray(image)
    emotion = emotion_detector(image)[0]
    return emotion['label'], emotion['score']

def analyze_emotion_video(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        label, score = analyze_emotion_image(frame)
        emotions.append(label)
        
        st.image(frame, channels="BGR")
        st.write(f"Detected Emotion: {label} (Score: {score:.2f})")

    cap.release()

if upload_type == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        label, score = analyze_emotion_image(np.array(image))
        st.write(f"Detected Emotion: {label} (Score: {score:.2f})")

elif upload_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        analyze_emotion_video("temp_video.mp4")
import streamlit as st
import cv2
from transformers import pipeline
from PIL import Image
import numpy as np

st.title("Emotion Detection System")
st.write("Upload an image or video to analyze emotions using Generative AI and LLMs.")
upload_type = st.radio("Choose input type:", ("Image", "Video"))

emotion_detector = pipeline("sentiment-analysis", model="mrm8488/t5-base-finetuned-emotion")

def analyze_emotion_image(image):
    image = Image.fromarray(image)
    emotion = emotion_detector(image)[0]
    return emotion['label'], emotion['score']

def analyze_emotion_video(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        label, score = analyze_emotion_image(frame)
        emotions.append(label)
        
        st.image(frame, channels="BGR")
        st.write(f"Detected Emotion: {label} (Score: {score:.2f})")

    cap.release()

if upload_type == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        label, score = analyze_emotion_image(np.array(image))
        st.write(f"Detected Emotion: {label} (Score: {score:.2f})")

elif upload_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        analyze_emotion_video("temp_video.mp4")
import streamlit as st
import cv2
from transformers import pipeline
from PIL import Image
import numpy as np

st.title("Emotion Detection System")
st.write("Upload an image or video to analyze emotions using Generative AI and LLMs.")
upload_type = st.radio("Choose input type:", ("Image", "Video"))

emotion_detector = pipeline("sentiment-analysis", model="mrm8488/t5-base-finetuned-emotion")

def analyze_emotion_image(image):
    image = Image.fromarray(image)
    emotion = emotion_detector(image)[0]
    return emotion['label'], emotion['score']

def analyze_emotion_video(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        label, score = analyze_emotion_image(frame)
        emotions.append(label)
        
        st.image(frame, channels="BGR")
        st.write(f"Detected Emotion: {label} (Score: {score:.2f})")

    cap.release()

if upload_type == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        label, score = analyze_emotion_image(np.array(image))
        st.write(f"Detected Emotion: {label} (Score: {score:.2f})")

elif upload_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        analyze_emotion_video("temp_video.mp4")
