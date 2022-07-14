import cv2
import torch


mask = cv2.imread("../Datasets/shoreline_ready_data/train/masks/IMG_1.png")
print(mask.shape)
mask = cv2.imread("../Datasets/shoreline_ready_data/train/masks/IMG_1.png", 1)
print(mask.shape)