import torch
import cv2
import numpy as np
import pyautogui

from model.unet_model import UNet
from utils.postprocess import heatmaps_to_keypoints

from sklearn.linear_model import LinearRegression

pyautogui.FAILSAFE = False

# TODO1: add more anchor points
ANCHOR_LOCATION = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
]


if __name__ == "__main__":

    # initialize model
    unet = UNet(n_channels=3, n_classes=8)

    # TODO: train on a larger resolution can help
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load("model_epoch_7.th", map_location=device)

    # initialize pyautogui
    screen_width, screen_height = pyautogui.size()

    # start calibration
    cam = cv2.VideoCapture(0)
    key = None

    anchor_detection = []
    # TODO: add more data point for each anchor location (different poses)
    for x, y in ANCHOR_LOCATION:
        coordinate_detection = []
        print("please look at the cursor location until it moves.")

        while len(coordinate_detection) < 5 or np.array(coordinate_detection).std(0).max() > 2:
            _, frame_orig = cam.read()
            pyautogui.moveTo(int(x * screen_width), int(y * screen_height))

            # TODO: use more augmentation during training for different scale/aspect_ratio scenarios.
            frame = cv2.resize(frame_orig, (320, 240))
            frame = frame.astype(np.float32)
            input = [torch.FloatTensor(frame.astype(np.float32)/255.0)]
            input = torch.stack(input)
            input = torch.permute(input, (0, 3, 1, 2))
            heatmap = model(input.to(device))
            heatmap = heatmap.detach().numpy()  # 1 x 8 x 180 x320

            keypoints = heatmaps_to_keypoints(heatmap, scale=2, threshold=0.3)

            if np.array(keypoints).min() == -1:
                continue

            coordinate_detection.append(keypoints)
            coordinate_detection = coordinate_detection[-5:]  # only save last 5 detections

            print(coordinate_detection)

        anchor_detection.append(np.array(coordinate_detection).mean(0).reshape(-1))

    anchor_detection = np.array(anchor_detection)  # 9 * 16
    anchor_location = np.array(ANCHOR_LOCATION)  # 9 * 2

    reg = LinearRegression().fit(anchor_detection, anchor_location)
    print(f"regression score: {reg.score(anchor_detection, anchor_location)}")

    # TODO: parallel processing for this part.
    # prediction
    while True:
        _, frame_orig = cam.read()

        frame = cv2.resize(frame_orig, (320, 240))
        frame = frame.astype(np.float32)
        input = [torch.FloatTensor(frame.astype(np.float32) / 255.0)]
        input = torch.stack(input)
        input = torch.permute(input, (0, 3, 1, 2))
        heatmap = model(input.to(device))
        heatmap = heatmap.detach().numpy()  # 1 x 8 x 180 x320

        keypoints = heatmaps_to_keypoints(heatmap, scale=2, threshold=0.3)

        if np.array(keypoints).min() == -1:
            continue

        detection = np.array(keypoints).reshape(-1)  # (16,)
        x, y = reg.predict([detection])[0]  # (2,)

        pyautogui.moveTo(int(screen_width * x), int(screen_height * y))

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
