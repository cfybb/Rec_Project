'''
from torch.utils.data import Dataset, DataLoader
class Facegaze(Dataset):
    def __init__(self)
'''
import sys
sys.path.append('C:/prdue/job_preperation_general/support_company/project/Rec_Project/data')
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os
from data_utils import data_augmentation
# initialization for hyperparameters
scale_factor = 4  # factor between input image and heatmap resolution.
sigma = 4  # sigma parameter for quadratic gaussian distribution heatmap
# add input size
input_x, input_y = 640, 360  # might change here.


class CustomDataset(Dataset):
    def __init__(self):
        # initialize
        self.data_paths = "C:/prdue/job_preperation_general/support_company/project/MPIIFaceGaze/annotationOverall.txt"
        self.folder_path = "C:/prdue/job_preperation_general/support_company/project/MPIIFaceGaze"
        self.data = self.read_txt(self.data_paths)

    def __len__(self):
        # return size
        return len(self.data)

    def __getitem__(self, idx):
        # find the correlated image and keypoint.
        line = self.data[idx]
        image_path, keypoints = line.split(' ', 1)
        image_id_path = os.path.join(self.folder_path, image_path)
        image = self.load_image(image_id_path)
        # find origional shape
        org_x, org_y = image.shape[1], image.shape[0]
        # print("org_x",org_x)
        # print("org_y",org_y)
        # ratio of the change.
        rat_x = input_x / org_x
        rat_y = input_y / org_y
        keypoints = [float(coord) for coord in keypoints.split()]
        # TODO: convert image/keypoints to input size
        image = cv2.resize(image, (input_x, input_y))
        keypoints = resize_keypoint_input(keypoints, rat_x, rat_y)

        # data augmentation (only change those not have -1)
        # print("super origion keypoints",keypoints)
        image, keypoints = data_augmentation(image, keypoints, options=["rescaling", "shifting", "rotation", "saturation", "random noise"])  # choose one as test.

        # return
        return image, keypoints

    def load_image(self, path):
        # return rbg image.

        image = cv2.imread(path)

        return (image)

    def read_txt(self, txt_path):
        with open(txt_path, 'r') as file:
            data = file.readlines()
        return data[1:]  # first line is empty
##########################################################################


def generate_gaussian_heatmap(x, y, image_shape, scale_factor, sigma=1.0):
    """
    gaussian heatmap
    """
    heatmap_size = (image_shape[0]//scale_factor, image_shape[1]//scale_factor)
    x, y = x/scale_factor, y/scale_factor
    x = np.clip(x, 0, heatmap_size[1] - 1)
    y = np.clip(y, 0, heatmap_size[0] - 1)

    x_coords = np.arange(0, heatmap_size[1], 1)
    y_coords = np.arange(0, heatmap_size[0], 1)
    X, Y = np.meshgrid(x_coords, y_coords)

    pos = np.dstack((X, Y))
    rv = multivariate_normal(mean=[x, y], cov=sigma)
    heatmap = rv.pdf(pos)

    # normalize
    heatmap = (heatmap / np.max(heatmap))*2*np.pi*sigma**2
    # print(heatmap.shape)
    return heatmap
###########################################################################


def collate_fn(batch):
    # image, label
    images, labels = zip(*batch)
    # print(size(images),size(labels))
    # min_pixel_value = min(torch.min(torch.FloatTensor(image)) for image in images)
    # max_pixel_value = max(torch.max(torch.FloatTensor(image)) for image in images)

    # print("Min pixel value:", min_pixel_value)
    # print("Max pixel value:", max_pixel_value)
    # data augmentation   not here
    # images, labels = data_augmentation(images, labels,options = ['rescaling', 'shifting']) #only do two, just in case.
    images = [torch.FloatTensor(image/255.0) for image in images]
    images = torch.stack(images)
    images = images.permute(0, 3, 1, 2)
    # cv2.imshow("image check", (255.0*images[0].permute(1, 2, 0).numpy()).astype(np.uint8))
    image_shape = (images.shape[2], images.shape[3])
    # print("Min pixel value:", images.min())
    # print("Max pixel value:", images.max())
    # deal with image

    # deal with label
    heatmaps = []  # save heatmap
    masks = []     # mask

    for label in labels:
        heatmap_per_sample = []
        mask_per_sample = []

        for i in range(0, len(label), 2):
            x, y = label[i], label[i + 1]
            # print("x",x)
            # print("y",y)
            if x == -1 and y == -1:
                # mask = 0 if -1 exist in the keypoint
                mask_per_sample.append(0)
                heatmap_per_sample.append(np.zeros((image_shape[0] // scale_factor, image_shape[1] // scale_factor)))  #append heatmap size
            else:
                mask_per_sample.append(1)
                # generate heatmap
                heatmap = generate_gaussian_heatmap(x, y, image_shape=image_shape, scale_factor=scale_factor)
                heatmap_per_sample.append(heatmap)

        heatmaps.append(heatmap_per_sample)
        masks.append(mask_per_sample)

    heatmaps = torch.tensor(np.array(heatmaps, dtype=np.float32))
    masks = torch.tensor(masks)
    masks = masks.unsqueeze(-1)
    masks = masks.unsqueeze(-1)

    return {'images': images, 'heatmaps': heatmaps, 'masks': masks}


def resize_keypoint_input(keypoints, rat_x, rat_y):
    return [keypoints[i] * rat_x if i % 2 == 0 and (keypoints[i] != -1 and keypoints[i+1] != -1) else
            (keypoints[i] * rat_y if (keypoints[i] != -1 and keypoints[i-1] != -1) else keypoints[i]) for i in range(len(keypoints))]


# validation check
if __name__ == "__main__":
    dataset = CustomDataset()
    # print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True, collate_fn=collate_fn)
    for batch in dataloader:
        image = (255.0*batch["images"][0].permute(1, 2, 0).numpy()).astype(np.uint8)         # TODO
        # print("max of image",np.max(image))
        # print("min of image", np.min(image))
        heatmap = batch["heatmaps"][0].permute(1, 2, 0).max(axis=-1).values.numpy()
        mask = batch["masks"][0]
        if mask[-1] == 0:
            continue
        # heatmap_vis = (heatmap * 255).astype(np.uint8)
        heatmap_vis = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        # print(heatmap_vis.shape)
        cv2.imshow("image", image)
        cv2.imshow("heatmap", heatmap_vis)
        key = cv2.waitKey(-1)
        if key == ord("q"):
            break
