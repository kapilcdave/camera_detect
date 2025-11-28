import cv2
import numpy as np
from ultralytics import YOLO

def get_neck_roi(keypoints, frame_shape):
    """Calculate neck region based on pose keypoints."""
    nose = keypoints[0]
    l_shoulder = keypoints[5]
    r_shoulder = keypoints[6]
    
    if not (np.any(l_shoulder) and np.any(r_shoulder)):
        return None
    
    shoulder_y = int(min(l_shoulder[1], r_shoulder[1]))
    shoulder_center_x = int((l_shoulder[0] + r_shoulder[0]) / 2)
    shoulder_width = int(abs(r_shoulder[0] - l_shoulder[0]))
    
    # Calculate chin position
    if np.any(nose):
        chin_y = int(nose[1] + (shoulder_y - nose[1]) * 0.48)
    else:
        chin_y = int(shoulder_y * 0.85)
    
    # Define ROI
    y_min = chin_y
    y_max = int(shoulder_y + (shoulder_y - chin_y) * 0.4)
    roi_width = int(shoulder_width * 0.5)
    x_min = int(shoulder_center_x - roi_width / 2)
    x_max = int(shoulder_center_x + roi_width / 2)
    
    # Validate
    if x_max <= x_min or y_max <= y_min or y_min < 0:
        return None
    
    return (x_min, y_min, x_max, y_max)

def detect_v_shape(roi):
    """Detect V-shape lines in the ROI."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Lower thresholds for more sensitive edge detection
    edges = cv2.Canny(blurred, 30, 90)
    
    # More lenient Hough Line parameters for easier detection
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=20, maxLineGap=10)
    
    if lines is None:
        return None, None
    
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Wider angle range to catch acute angles (sharper V-shapes)
        if -75 < angle < -15:  # Left side of V
            left_lines.append((line[0], length))
        elif 15 < angle < 75:  # Right side of V
            right_lines.append((line[0], length))
    
    if not left_lines or not right_lines:
        return None, None
    
    # Get longest line from each side
    best_left = max(left_lines, key=lambda x: x[1])[0]
    best_right = max(right_lines, key=lambda x: x[1])[0]
    
    # Check if they form a valid V (similar vertical position)
    left_center_y = (best_left[1] + best_left[3]) / 2
    right_center_y = (best_right[1] + best_right[3]) / 2
    
    if abs(left_center_y - right_center_y) > roi.shape[0] * 0.4:
        return None, None
    
    return best_left, best_right

def main():
    # Initialize
    model = YOLO('yolov8n-pose.pt')
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Quarter Zip Detector running. Press 'q' to quit.")
    
    detection_buffer = []
    buffer_size = 25
    threshold = 0.6
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run pose detection
        results = model(frame, verbose=False)
        display_frame = frame.copy()
        
        # Get keypoints
        keypoints = results[0].keypoints.xy.cpu().numpy()
        detected = False
        
        if len(keypoints) > 0:
            # Get neck ROI
            roi_coords = get_neck_roi(keypoints[0], frame.shape)
            
            if roi_coords:
                x_min, y_min, x_max, y_max = roi_coords
                roi = frame[y_min:y_max, x_min:x_max]
                
                # Detect V-shape
                left_line, right_line = detect_v_shape(roi)
                
                if left_line is not None and right_line is not None:
                    detected = True
                    
                    # Draw V-shape lines with thicker lines for better visibility
                    x1, y1, x2, y2 = left_line
                    cv2.line(display_frame, (x_min + x1, y_min + y1), 
                             (x_min + x2, y_min + y2), (0, 255, 0), 5)
                    
                    x1, y1, x2, y2 = right_line
                    cv2.line(display_frame, (x_min + x1, y_min + y1), 
                             (x_min + x2, y_min + y2), (0, 255, 0), 5)
                
                # Draw neck area box
                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                cv2.putText(display_frame, "NECK AREA", (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Update detection buffer
        detection_buffer.append(1 if detected else 0)
        if len(detection_buffer) > buffer_size:
            detection_buffer.pop(0)
        
        # Stable detection
        stable = sum(detection_buffer) / len(detection_buffer) >= threshold if detection_buffer else False
        
        # Display result
        if stable:
            cv2.putText(display_frame, "QUARTER ZIP DETECTED!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            cv2.putText(display_frame, "No Quarter Zip", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Quarter Zip Detector (YOLOv8)', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
