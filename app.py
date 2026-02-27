import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

st.set_page_config(
    page_title="Fruit Detection App",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎🍌🍊 Fruit Object Detection")
st.write("Upload an image to detect apples, bananas, and oranges!")

@st.cache_resource
def load_model():
    model = YOLO('fruit_detection_model.pt', task='detect')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner('Detecting fruits...'):
        results = model.predict(image, conf=0.25, device='cpu', imgsz=320)
        annotated_image = results[0].plot()
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("Detection Results")
        st.image(annotated_image, use_container_width=True)

    st.subheader("📊 Detection Details")
    detections = results[0].boxes

    if len(detections) > 0:
        class_names = ['apple', 'banana', 'orange']
        for i, box in enumerate(detections):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"**Object {i+1}:** {class_names[cls].capitalize()} - Confidence: {conf:.2%}")
    else:
        st.write("No fruits detected in the image.")
