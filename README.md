# Quarter Zip Detector 🧥

Real-time computer vision application that detects whether someone is wearing a quarter zip using YOLOv8 pose estimation and OpenCV edge detection.

## Features

- **Real-time Detection**: Analyzes webcam feed in real-time
- **YOLOv8 Pose Estimation**: Uses state-of-the-art pose detection to locate the neck area
- **V-Shape Recognition**: Detects the characteristic V-shaped collar of quarter zips
- **Temporal Stability**: Smoothed detection to prevent false positives
- **Visual Feedback**: Shows detection area and V-shape lines when quarter zip is detected

## Demo

The detector identifies the neck area and looks for two diagonal lines forming a V-shape - the signature feature of a quarter zip collar.

## How It Works

1. **Pose Detection**: YOLOv8 identifies key body points (nose, shoulders)
2. **ROI Extraction**: Calculates neck region based on facial and shoulder landmarks
3. **Edge Detection**: Uses Canny edge detection to find lines in the neck area
4. **V-Shape Matching**: Identifies two diagonal lines (30-65° angles) forming a V
5. **Temporal Filtering**: Averages detection over 25 frames for stable output

## Installation

### Prerequisites

- Python 3.8+
- Webcam

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd camera_detect
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install opencv-python ultralytics numpy
   ```

## Usage

Run the detector:

```bash
python quarter_zip_detector.py
```

Or using the virtual environment directly:

```bash
./venv/bin/python quarter_zip_detector.py
```

**Controls:**
- Press `q` to quit

## Configuration

You can adjust detection parameters in the code:

- `buffer_size`: Number of frames to average (default: 25)
- `threshold`: Detection confidence threshold (default: 0.6)
- Edge detection parameters in `detect_v_shape()` function

## Technical Details

**Technology Stack:**
- **YOLOv8**: Pose estimation for body keypoint detection
- **OpenCV**: Image processing and edge detection
- **NumPy**: Numerical operations

**Detection Algorithm:**
1. Locate neck region using nose and shoulder keypoints
2. Apply Gaussian blur and Canny edge detection
3. Use Hough Line Transform to detect line segments
4. Filter lines by angle (-65° to -25° for left, 25° to 65° for right)
5. Verify V-shape formation by checking vertical alignment
6. Apply temporal smoothing for stable results

## Requirements

```
opencv-python>=4.8.0
ultralytics>=8.0.0
numpy>=1.24.0
```

## Limitations

- Requires good lighting conditions
- Quarter zip zipper must be visible
- Single person detection (analyzes first detected person)
- May trigger on other V-neck clothing

## Future Improvements

- [ ] Multi-person detection
- [ ] Distinguish between quarter zips and other V-neck garments
- [ ] Add clothing color detection
- [ ] Support for video file input
- [ ] Performance optimization for lower-end hardware

## License

MIT License - feel free to use and modify!

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
