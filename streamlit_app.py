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

    st.write("Take a photo to check for a quarter zip!")
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # Convert to CV2 format
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Process
        processed_frame, is_detected = detector.process_frame(cv2_img)
        
        # Display Result
        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Processed Image")
        
        if is_detected:
            st.success("Quarter Zip Detected! ✅")
        else:
            st.warning("No Quarter Zip Detected.")

st.sidebar.markdown("### Debug Info")
st.sidebar.info("Model: YOLOv8-pose")
