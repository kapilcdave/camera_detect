import streamlit as st
import cv2
import numpy as np
from quarter_zip_detector import QuarterZipDetector
from PIL import Image

st.set_page_config(page_title="Quarter Zip Detector", layout="wide")

st.title("Quarter Zip Detector 🔍")
st.text("Detects if you are wearing a quarter zip using Pose Estimation and V-shape analysis.")

# Initialize the detector
@st.cache_resource
def load_detector():
    return QuarterZipDetector()

detector = load_detector()

# Option to use camera
use_webcam = st.checkbox("Use Webcam")

if use_webcam:

    run_detection = st.checkbox("Run Continuous Detection (Local Only)")
    
    frame_window = st.image([])
    
    if run_detection:
        cap = cv2.VideoCapture(0)
        
        stop_button = st.button("Stop")
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Could not access camera.")
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process
            processed_frame, is_detected = detector.process_frame(frame)
            
            # Update image
            frame_window.image(processed_frame)
            
        cap.release()

st.sidebar.markdown("### Debug Info")
st.sidebar.info("Model: YOLOv8-pose")
