import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import time
from ultralytics import YOLO

class QuarterZipDetector:
    def __init__(self, model_path='yolov8n-pose.pt', camera_id=0):
        self.model = YOLO(model_path)
        self.camera_id = camera_id
        
        self.conf_threshold = 0.5
        self.min_line_length = 20
        self.max_line_gap = 10
        
    def get_neck_roi(self, keypoints, frame_shape):
        """
        Extracts the Region of Interest (ROI) around the neck/upper chest area.
        Returns: (x1, y1, x2, y2) or None
        """
        # Keypoints: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, ...]
        # We need shoulders (5, 6) and nose (0) can help for height reference
        
        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]
        nose = keypoints[0]
        
        # Check confidence (if x,y are 0 or conf is low, usually filtered by YOLO, but checking non-zero helps)
        if hasattr(l_shoulder, 'conf') and l_shoulder.conf < 0.5: return None
        if hasattr(r_shoulder, 'conf') and r_shoulder.conf < 0.5: return None
        if l_shoulder[0] == 0 or r_shoulder[0] == 0: return None

        # Calculate bounding box for neck area
        # Center x is mid-shoulder
        sx_min = min(l_shoulder[0], r_shoulder[0])
        sx_max = max(l_shoulder[0], r_shoulder[0])
        sy_min = min(l_shoulder[1], r_shoulder[1])
        
        shoulder_width = abs(sx_max - sx_min)
        shoulder_center_x = (sx_min + sx_max) / 2
        shoulder_y = sy_min # approximate shoulder height
        
        # If nose is detected, use it as upper bound, else estimate
        if nose[0] != 0:
            top_y = nose[1] + (shoulder_y - nose[1]) * 0.5 # Halfway between nose and shoulder
        else:
            top_y = shoulder_y - shoulder_width * 0.3 # Rough estimate

        bottom_y = shoulder_y + shoulder_width * 0.4 # A bit down the chest
        
        # ROI Width: narrower than full shoulder width to focus on collar
        roi_w = shoulder_width * 0.5
        
        x1 = int(shoulder_center_x - roi_w/2)
        x2 = int(shoulder_center_x + roi_w/2)
        y1 = int(top_y)
        y2 = int(bottom_y)
        
        # Clip to frame
        h, w = frame_shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        return (x1, y1, x2, y2)

    def detect_v_shape(self, roi):
        """
        Detects V-shaped lines in the ROI.
        Returns: Tuple of (line_left, line_right) or (None, None)
        Lines are (x1, y1, x2, y2) in ROI coordinates.
        """
        if roi.size == 0: return None, None
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Blur to reduce noise (texture of fabric)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Probabilistic Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                                minLineLength=self.min_line_length, 
                                maxLineGap=self.max_line_gap)
        
        if lines is None:
            return None, None
            
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Ensure y2 > y1 for consistency (points downwards)
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
                
            dx = x2 - x1
            dy = y2 - y1
            
            if dy == 0: continue # Horizontal line, ignore
            
            # Angle in degrees. 
            # dx = 0 -> vertical (90 deg). 
            # \ shape: dx > 0. Angle approx 60-80 deg from horizontal
            # / shape: dx < 0. Angle approx 100-120 deg from horizontal (or -60 to -80)
            
            # Calculate angle relative to vertical for easier mental model?
            # Or just use slopes. 
            
            slope = dx / dy # Run / Rise (inverse slope) because we want near vertical lines
            
            # Normal slope (m = dy/dx). 
            # If |m| is high, it's vertical. If |m| is low, it's horizontal.
            # We want steep lines, so |m| > 1 roughly.
            
            # Left side of V (\): x increases as y increases. dx > 0.
            # Right side of V (/): x decreases as y increases. dx < 0.
            
            # Check for steepness. Let's say we want between 30 and 80 degrees from horizontal.
            angle_deg = np.degrees(np.arctan2(dy, abs(dx)))
            
            if not (30 < angle_deg < 85):
                continue
                
            length = np.sqrt(dx**2 + dy**2)
            
            if dx > 0: # \ shape (Left side of V visually, but on the right of the person?)
                # Wait: \ is left side of the letter V.
                left_lines.append((line[0], length))
            else: # / shape (Right side of V)
                right_lines.append((line[0], length))
        
        if not left_lines or not right_lines:
            return None, None
            
        # Get the longest lines for now
        # Ideally we'd check if they meet at the bottom
        best_left = max(left_lines, key=lambda x: x[1])[0]
        best_right = max(right_lines, key=lambda x: x[1])[0]
        
        # Geometric validation: Left line center should be to the left of Right line center
        l_cx = (best_left[0] + best_left[2])/2
        r_cx = (best_right[0] + best_right[2])/2
        
        if l_cx > r_cx:
            # Crossed lines or wrong identification
            return None, None
            
        return best_left, best_right

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return

        print("Quarter Zip Detector started. Press 'q' to quit.")
        
        prev_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Run Pose Estimation
            results = self.model(frame, verbose=False, stream=True)
            
            # Process results (generator)
            for result in results:
                keypoints = result.keypoints.xy.cpu().numpy() # [N, 17, 2]
                
                for kpts in keypoints:
                    if len(kpts) == 0: continue
                    
                    roi_coords = self.get_neck_roi(kpts, frame.shape)
                    
                    if roi_coords:
                        x1, y1, x2, y2 = roi_coords
                        
                        # Draw ROI (optional, for debug)
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        
                        roi = frame[y1:y2, x1:x2]
                        l_line, r_line = self.detect_v_shape(roi)
                        
                        if l_line is not None and r_line is not None:
                            # Convert back to global coordinates and draw
                            lx1, ly1, lx2, ly2 = l_line
                            rx1, ry1, rx2, ry2 = r_line
                            
                            cv2.line(frame, (x1+lx1, y1+ly1), (x1+lx2, y1+ly2), (0, 255, 0), 3)
                            cv2.line(frame, (x1+rx1, y1+ry1), (x1+rx2, y1+ry2), (0, 255, 0), 3)
                            
                            # Add label
                            cv2.putText(frame, "QUARTER ZIP", (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow('Quarter Zip Detector', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = QuarterZipDetector()
    detector.run()
