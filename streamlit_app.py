import streamlit as st
import cv2
import numpy as np
from quarter_zip_detector import QuarterZipDetector
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Quarter Zip Detector", layout="wide")

st.title("Quarter Zip Detector 🔍")
st.text("Live detection for quarter zips using Pose Estimation.")

# Initialize global resources
@st.cache_resource
def get_detector():
    return QuarterZipDetector()

# Define the processor class for webrtc
class QuarterZipProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = get_detector()

    def recv(self, frame):
        # Convert AV frame to NumPy array (BGR)
        img = frame.to_ndarray(format="bgr24")
        
        # Process the frame using our existing logic
        # process_frame returns (annotated_frame, is_detected)
        annotated_image, _ = self.detector.process_frame(img)
        
        # Return the new frame
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

st.write("Click 'Start' to enable the camera.")
webrtc_streamer(key="quarter-zip", video_processor_factory=QuarterZipProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True)

st.sidebar.markdown("### Debug Info")
st.sidebar.info("Model: YOLOv8-pose")

