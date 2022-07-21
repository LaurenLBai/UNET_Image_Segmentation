
from utils import(visualize,get_preprocessing, print_examples)
import cv2 
import os
from dataset import Dataset
import segmentation_models_pytorch as smp

# file1 = "../Datasets/shoreline_ready_data/val/images/IMG_2.png"
# file2 = "../Datasets/shoreline_ready_data/val/masks/IMG_2.png"

# file1 = "../Datasets/Shoreline_Test_Dataset/mask_patches/p50@1@18.png"
# file2 = "../Datasets/Shoreline_Test_Dataset/output/pred_18.png"

file1 = "../Datasets/Shoreline_Test_Dataset/images/p50.JPG"
file2 = "../Datasets/Shoreline_Test_Dataset/masks/p50.png"


image1 = cv2.imread(file1, 1)
print(image1.shape)


image2 = cv2.imread(file2, 0)
print(image2.shape)


visualize(image1 = image1, image2 = image2)



# TEST_IMG_DIR = "../Datasets/Shoreline_Test_Dataset/image_patches/"
# TEST_MASK_DIR = "../Datasets/Shoreline_Test_Dataset/mask_patches/"

# TEST_IMG_DIR =  "../Datasets/Shoreline_Test_Dataset/image_patches/"
# TEST_MASK_DIR = "../Datasets/Shoreline_Test_Dataset/output/"

# for i in range(0, 5):
#     print_examples(TEST_IMG_DIR, TEST_MASK_DIR)



# preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet34', 'imagenet')

# augmented_dataset = Dataset(
#     TEST_IMG_DIR, 
#     TEST_MASK_DIR, 
#     classes=['water'],
# )

# # same image with different random transforms
# for i in range(21, 25):
#     image, mask = augmented_dataset[i]
#     print(image.shape)
#     print(mask.shape)
#     # mask=mask.squeeze(-1)
#     # print(mask.shape)
#     visualize(image=image, mask=mask)

