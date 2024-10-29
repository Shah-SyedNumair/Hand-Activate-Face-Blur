# Hand and Face Detection with Blurring

This project uses Python to detect both hands and faces in real-time video. It blurs the face when the tips of the middle finger and thumb of either hand touch, and unblurs it when they are apart.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- OpenCV (`cv2`)
- MediaPipe

You can install the required packages using pip:
pip install opencv-python mediapipe

## Usage

1. Run Script:
   python hand_face_blur.py

2. Allow the script to access your webcam. The program will open a window showing the video feed, where it will detect hands and faces.

3. When the tips of the middle finger and thumb touch, the detected face will be blurred. When they are apart, the face will return to normal.

## Code Overview
The main components of the script are:

Hand Detection: Using MediaPipe to track the hand landmarks.
Face Detection: Using MediaPipe to detect faces.
Blurring Logic: Implementing a condition to blur/unblur the face based on the distance between the tips of the fingers.
## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- OpenCV
- MediaPipe
