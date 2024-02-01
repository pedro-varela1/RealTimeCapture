# Check camera with cv2
import cv2

# Check available cameras
n_cameras = 8   # Number of cameras to check

for i in range(n_cameras):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"Camera {i} is not available")
    else:
        print(f"Camera {i} is available")
        cap.release()