
from utils import(visualize,get_preprocessing, print_examples)
import cv2 
import os
from dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
import torch

# file1 = "../Datasets/shoreline_ready_data/val/images/IMG_2.png"
# file2 = "../Datasets/shoreline_ready_data/val/masks/IMG_2.png"

# file1 = "../Datasets/Shoreline_Test_Dataset/mask_patches/p50@1@18.png"
# file2 = "../Datasets/Shoreline_Test_Dataset/output/pred_18.png"

# file1 = "../Datasets/Shoreline_Test_Dataset/images/p50.JPG"
# file2 = "../Datasets/Shoreline_Test_Dataset/masks/p50.png"


# image1 = cv2.imread(file1, 1)
# print(image1.shape)


# image2 = cv2.imread(file2, 0)
# print(image2.shape)


# visualize(image1 = image1, image2 = image2)



TEST_IMG_DIR = "../Datasets/Shoreline_Test_Dataset/image_patches/"
TEST_MASK_DIR = "../Datasets/Shoreline_Test_Dataset/mask_patches/"
CLASSES = ['water']
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'
LOSS = smp.utils.losses.DiceLoss()
METRICS = [
    smp.utils.metrics.IoU(threshold = 0.5),
    #smp.utils.metrics.Accuracy(threshold = 0.5),
    ]

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

best_model = torch.load('best_model.pth')
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
# test dataset without transformations for image visualization
test_dataset = Dataset(
    TEST_IMG_DIR, 
    TEST_MASK_DIR, 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
test_dataloader = DataLoader(test_dataset)
# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=LOSS,
    metrics=METRICS,
    device=DEVICE,
)

# logs = test_epoch.run(test_dataloader)

test_dataset_vis = Dataset(
    TEST_IMG_DIR, TEST_MASK_DIR, 
    classes=CLASSES,
)
for i in range(2):
    n = np.random.choice(len(test_dataset))
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    print("image shape: ", image.shape)
    gt_mask = gt_mask.squeeze()
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    print("shape before prediction: ", x_tensor.shape)
    pr_mask = best_model.predict(x_tensor)
    print("output shape: ", pr_mask.shape)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    print("output shape after squeeze and numpy: ", pr_mask.shape)

    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask
    )