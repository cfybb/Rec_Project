import cv2
import numpy as np
import torch

def camera_stream_to_tensor():
    """
    Function to capture camera stream and convert to tensor.
    """
    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return None

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is captured
    if not ret:
        print("Error: Unable to capture frame.")
        cap.release()
        return None

    # Convert frame to tensor
    tensor = torch.from_numpy(np.transpose(frame, (2, 0, 1))).float()

    # Release the camera
    cap.release()

    return tensor, frame


# Test the function
# camera_tensor = camera_stream_to_tensor()
# print("Camera tensor shape:", camera_tensor.shape)