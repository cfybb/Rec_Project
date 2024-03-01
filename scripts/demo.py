import torch
import cv2
import numpy as np

from model.unet_model import UNet
from utils.postprocess import heatmaps_to_keypoints


if __name__ == "__main__":
    unet = UNet(n_channels=3, n_classes=8)
    model = torch.load("model_epoch_9.th", map_location=torch.device("cpu"))

    cam = cv2.VideoCapture(0)
    while True:
        _, frame_orig = cam.read()
        frame = cv2.resize(frame_orig, (320, 180))
        frame = frame.astype(np.float32)
        input = [torch.FloatTensor(frame.astype(np.float32)/255.0)]
        input = torch.stack(input)
        input = torch.permute(input, (0, 3, 1, 2))
        heatmap = model(input)
        heatmap = heatmap.detach().numpy()  # 1 x 8 x 180 x320

        # heatmap = heatmap[0].max(0)  # 180, 320
        # heatmap = (heatmap * 255).astype(np.uint8)
        # heatmap_vis = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #
        # frame_vis = (frame * 0.5 + heatmap_vis * 0.5).astype(np.uint8)


        if heatmap[0][0].max() > 0.6:
            keypoints = heatmaps_to_keypoints(heatmap, scale=4, threshold=0.3)
            for x, y in keypoints:
                if x < 0 or y < 0:
                    continue
                cv2.circle(frame_orig, (int(x), int(y)), 3, (0, 255, 0), -1)

        cv2.imshow("heatmap", frame_orig)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
