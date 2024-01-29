'''
from torch.utils.data import Dataset, DataLoader
class Facegaze(Dataset):
    def __init__(self)
'''
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os

class CustomDataset(Dataset):
    def __init__(self, data_paths):
        # 在这里初始化你的数据路径等信息
        self.data_paths = "C:\prdue\job_preperation_general\support_company\project\MPIIFaceGaze\annotationOverall.txt"
        self.folder_path = "C:\prdue\job_preperation_general\support_company\project\MPIIFaceGaze"
        self.data = self.read_txt(data_paths)

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引加载图像和标签数据
        line = self.data[idx]
        image_path, keypoints = line.split(' ', 1)
        image_id_path = os.path.join(self.folder_path,image_path)
        image = self.load_image(image_path)
        keypoints = [float(coord) for coord in keypoints.split()]

        # 返回图像和关键点坐标
        return image, keypoints

    def load_image(self, path):
        # 实现加载图像的逻辑
        # 返回图像的张量表示

        image = cv2.imread(path)

        return(image)

    def read_txt(self,txt_path):
        with open(txt_path, 'r') as file:
            data = file.readlines()
        return data

def generate_gaussian_heatmap(x, y, heatmap_size=(64, 36), sigma=1.0):
    """
    生成高斯分布的热图
    """
    x, y = int(x), int(y)
    x = np.clip(x, 0, heatmap_size[1] - 1)
    y = np.clip(y, 0, heatmap_size[0] - 1)

    x_coords = np.arange(0, heatmap_size[1], 1)
    y_coords = np.arange(0, heatmap_size[0], 1)
    X, Y = np.meshgrid(x_coords, y_coords)

    pos = np.dstack((X, Y))
    rv = multivariate_normal(mean=[x, y], cov=sigma)
    heatmap = rv.pdf(pos)

    # 归一化
    heatmap = heatmap / np.max(heatmap)
    return heatmap

def collate_fn(batch):
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]

    # 处理图像
    images = torch.stack(images)

    # 处理标签
    heatmaps = []  # 用于存储热图
    masks = []     # 用于存储掩码

    for label in labels:
        heatmap_per_sample = []
        mask_per_sample = []

        for i in range(0, len(label), 2):
            x, y = label[i], label[i + 1]
            if x == -1 and y == -1:
                # 如果关键点坐标为-1，表示无效，设置相应的mask为0
                mask_per_sample.append(0)
                heatmap_per_sample.append(np.zeros((64, 36)))  # 假设热图大小为 (64, 36)
            else:
                mask_per_sample.append(1)
                # 使用高斯分布生成关键点的热图
                heatmap = generate_gaussian_heatmap(x, y)
                heatmap_per_sample.append(heatmap)

        heatmaps.append(heatmap_per_sample)
        masks.append(mask_per_sample)

    heatmaps = torch.tensor(heatmaps)
    masks = torch.tensor(masks)

    return {'images': images, 'heatmaps': heatmaps, 'masks': masks}



