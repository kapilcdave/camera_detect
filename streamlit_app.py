import streamlit as st
import cv2
import numpy as np
from quarter_zip_detector import QuarterZipDetector

st.set_page_config(page_title="Quarter Zip Detector", layout="wide")

st.title("Quarter Zip Detector 🔍")
st.text("Detects if you are wearing a quarter zip using Pose Estimation.")

@st.cache_resource
def get_detector():
    return QuarterZipDetector()

detector = get_detector()

st.write("Take a photo to check for a quarter zip!")
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    processed_frame, is_detected = detector.process_frame(cv2_img)
    
    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption="Processed Image")
    
    if is_detected:
        st.success("Quarter Zip Detected! ✅")
    else:
        st.warning("No Quarter Zip Detected.")

st.markdown("---")
st.subheader("Download Local Script")
st.text("You can run this detector locally on your machine for a live video feed.")

with open("quarter_zip_detector.py", "rb") as file:
    st.download_button(
        label="Download quarter_zip_detector.py",
        data=file,
        file_name="quarter_zip_detector.py",
        mime="text/x-python"
    )

st.sidebar.markdown("### Debug Info")
st.sidebar.info("Model: YOLOv8-pose")
