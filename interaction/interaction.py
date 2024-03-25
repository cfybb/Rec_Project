import torch
import cv2
import os
import numpy as np
import sys

sys.path.append('C:/prdue/job_preperation_general/support_company/project/Rec_Project/utils')
sys.path.append('C:/prdue/job_preperation_general/support_company/project/Rec_Project')
from model.unet_model import UNet
from utils.postprocess import heatmaps_to_keypoints






if __name__ == "__main__":
    folder_path_FG = "C:/prdue/job_preperation_general/support_company/project/MPIIFaceGaze"
    save_file_path = "C:/prdue/job_preperation_general/support_company/project/MPIIFaceGaze"
    save_file_name = "annotationOverall.txt"
    save_file_path_FG = os.path.join(save_file_path, save_file_name)
    input_data = []
    unet = UNet(n_channels=3, n_classes=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load("model_epoch_7.th", map_location=device)

    # load pictures and retrain for the pupil.
    with open(save_file_path_FG,'r') as file:
        picture_info = file.readlines()
    # go thorugh the picture and use the model to train the pupil.
    print(picture_info[0])
    for p in range(len(picture_info)):
        # get the file directory
        data = picture_info[p].split()
        pic_dir = data[0]
        # get the p folder dir.
        pxx = pic_dir.split("/")[0]
        rest = pic_dir[len(pxx) + 1:]
        # get the screen axis.
        get_scrn_dir = os.path.join(folder_path_FG,pxx,pxx+".txt")
        with open(get_scrn_dir,'r') as file:
            full_pic_dir = file.readlines()
        for j in range(len(full_pic_dir)):
            out_data = full_pic_dir[j].split()
            if out_data[0] == rest:
                scrn_axis = [out_data[1],out_data[2]]

        # print(pic_dir)
        # get image
        figure_dir = os.path.join(folder_path_FG,pic_dir)
        img = cv2.imread(figure_dir)
        # put it into the model.
        img = cv2.resize(img, (320, 240))
        img = img.astype(np.float32)
        img = [torch.FloatTensor(img.astype(np.float32)/255.0)]
        img = torch.stack(img)
        img = torch.permute(img, (0, 3, 1, 2))
        heatmap = model(img.to(device))
        heatmap = heatmap.detach().cpu().numpy()
        if heatmap.max() > 0.4:
            keypoints = heatmaps_to_keypoints(heatmap, scale=2, threshold=0.3)
        # save
        input_data.append((keypoints,scrn_axis))


    #linear regression.
    # Extracting input and output from input_data
    inputs = np.array([data[0] for data in input_data])
    outputs = np.array([data[1] for data in input_data])

    # Reshape inputs and outputs if needed (e.g., from a list of vectors to a 2D array)
    inputs = inputs.reshape(-1, 1)  # Assuming each input is a single value
    outputs = outputs.reshape(-1,1)

    # Perform linear regression
    matrix = LinearRegression().fit(inputs, outputs)

    # Get the coefficients (transfer matrix) and intercept
    coefficients = matrix.coef_
    intercept = matrix.intercept_




