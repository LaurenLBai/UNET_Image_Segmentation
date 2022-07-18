from utils import(visualize)
import cv2 
import matplotlib.pyplot as plt

file1 = "../Datasets/Shoreline_Test_Dataset/mask_patches/p50@1@1.png"
file2 = "../Datasets/Shoreline_Test_Dataset/image_patches/p50@1@1.JPG"

# visualize(image = file1, mask = file2)
image = cv2.imread(file1)
print(image.shape)

image2 = cv2.imread(file2)
print(image2.shape)


visualize(image1 = image, image2 = image2)

