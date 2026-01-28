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
    
        self.alpha = 0.2 
        self.prev_left = None
        self.prev_right = None
        self.locked_on_counter = 0
        self.lock_threshold = 5 
        
    def get_neck_roi(self, keypoints, frame_shape):
        """
        Extracts the Region of Interest (ROI) around the neck/upper chest area.
        Returns: (x1, y1, x2, y2) or None
        """
      
        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]
        nose = keypoints[0]
        
        if hasattr(l_shoulder, 'conf') and l_shoulder.conf < 0.5: return None
        if hasattr(r_shoulder, 'conf') and r_shoulder.conf < 0.5: return None
        if l_shoulder[0] == 0 or r_shoulder[0] == 0: return None

        sx_min = min(l_shoulder[0], r_shoulder[0])
        sx_max = max(l_shoulder[0], r_shoulder[0])
        sy_min = min(l_shoulder[1], r_shoulder[1])
        
        shoulder_width = abs(sx_max - sx_min)
        shoulder_center_x = (sx_min + sx_max) / 2
        shoulder_y = sy_min 

        if nose[0] != 0:
            top_y = nose[1] + (shoulder_y - nose[1]) * 0.5 
        else:
            top_y = shoulder_y - shoulder_width * 0.3   

        bottom_y = shoulder_y + shoulder_width * 0.4 
      
        roi_w = shoulder_width * 0.6 
        
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
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                                minLineLength=self.min_line_length, 
                                maxLineGap=self.max_line_gap)
        
        if lines is None:
            return None, None
            
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
                
            dx = x2 - x1
            dy = y2 - y1
            
            if dy == 0: continue 
            
            slope = dx / dy 
            angle_deg = np.degrees(np.arctan2(dy, abs(dx)))
            
            # angle range for V-shape 
            if not (25 < angle_deg < 85): 
                continue
                
            length = np.sqrt(dx**2 + dy**2)
            
            if dx > 0: 
                left_lines.append((line[0], length))
            else: 
                right_lines.append((line[0], length))
        
        if not left_lines or not right_lines:
            return None, None
            
        best_left = max(left_lines, key=lambda x: x[1])[0]
        best_right = max(right_lines, key=lambda x: x[1])[0]
        
        l_cx = (best_left[0] + best_left[2])/2
        r_cx = (best_right[0] + best_right[2])/2
        
        if l_cx > r_cx:
            return None, None
            
        return best_left, best_right

    def smooth_line(self, current, prev):
        """Applies Exponential Moving Average to line coordinates."""
        if prev is None:
            return current
        
        smoothed = []
        for c, p in zip(current, prev):
            val = self.alpha * c + (1 - self.alpha) * p
            smoothed.append(val)
        return np.array(smoothed, dtype=int)

    def intersect_lines(self, l_line, r_line):
        """Finds intersection point of two lines."""
        x1, y1, x2, y2 = l_line
        x3, y3, x4, y4 = r_line

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0: return None
        
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        
        return int(px), int(py)

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
            
            detected_this_frame = False
            
            results = self.model(frame, verbose=False, stream=True)
            
            for result in results:
                keypoints = result.keypoints.xy.cpu().numpy() 
                
                for kpts in keypoints:
                    if len(kpts) == 0: continue
                    
                    roi_coords = self.get_neck_roi(kpts, frame.shape)
                    
                    if roi_coords:
                        rx, ry, rx2, ry2 = roi_coords
                        roi_w = rx2 - rx
                        roi_h = ry2 - ry
                        
                        roi = frame[ry:ry2, rx:rx2]
                        l_line_raw, r_line_raw = self.detect_v_shape(roi)
                        
                        if l_line_raw is not None and r_line_raw is not None:
                            l_line_global = [l_line_raw[0]+rx, l_line_raw[1]+ry, l_line_raw[2]+rx, l_line_raw[3]+ry]
                            r_line_global = [r_line_raw[0]+rx, r_line_raw[1]+ry, r_line_raw[2]+rx, r_line_raw[3]+ry]

                            self.prev_left = self.smooth_line(l_line_global, self.prev_left)
                            self.prev_right = self.smooth_line(r_line_global, self.prev_right)
                            
                            detected_this_frame = True
                            self.locked_on_counter = self.lock_threshold 
                            
                        elif self.locked_on_counter > 0 and self.prev_left is not None:
                             self.locked_on_counter -= 1
                             detected_this_frame = True
                        else:
                            self.locked_on_counter = 0
                            self.prev_left = None 
                            self.prev_right = None

                        # draw lines
                        if detected_this_frame and self.prev_left is not None and self.prev_right is not None:     
                            l_curr = self.prev_left
                            r_curr = self.prev_right
                            
                            intersect = self.intersect_lines(l_curr, r_curr)
                            
                            if intersect:
                                ix, iy = intersect
                                
                                l_top = (l_curr[0], l_curr[1]) if l_curr[1] < l_curr[3] else (l_curr[2], l_curr[3])
                                r_top = (r_curr[0], r_curr[1]) if r_curr[1] < r_curr[3] else (r_curr[2], r_curr[3])
                                
                                cv2.line(frame, l_top, (ix, iy), (0, 255, 0), 4, cv2.LINE_AA)
                                cv2.line(frame, r_top, (ix, iy), (0, 255, 0), 4, cv2.LINE_AA)
                                   
                                cv2.circle(frame, (ix, iy), 6, (0, 200, 0), -1)
                                
                                text = "QUARTER ZIP DETECTED"
                                t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                                tx = int((rx + rx2)/2 - t_size[0]/2)
                                ty = ry - 15
                                
                                cv2.rectangle(frame, (tx-5, ty-t_size[1]-5), (tx+t_size[0]+5, ty+5), (0,0,0), -1)
                                cv2.putText(frame, text, (tx, ty), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            cv2.imshow('Quarter Zip Detector', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = QuarterZipDetector()
    detector.run()
