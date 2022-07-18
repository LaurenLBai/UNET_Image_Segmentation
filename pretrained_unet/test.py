import torch
import torchvision
from utils import (
    get_preprocessing,
    visualize,
    
)
from dataset import Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np
from tqdm import tqdm


TEST_IMG_DIR = "../Datasets/Shoreline_Test_Dataset/image_patches/"
TEST_MASK_DIR = "../Datasets/Shoreline_Test_Dataset/mask_patches/"
OUTPUT_DIR = "../Datasets/Shoreline_Test_Dataset/output/"
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['water']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def main():
    file = "../Datasets/shoreline_ready_data/val/images/"
    # load best saved checkpoint
    best_model = torch.load('best_model.pth')
    # create test dataset
    test_dataset = Dataset(
        TEST_IMG_DIR, 
        TEST_MASK_DIR, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # test_dataloader = DataLoader(test_dataset)

    # # evaluate model on test set
    # loss = smp.utils.losses.DiceLoss()
    # metrics = [
    #     smp.utils.metrics.IoU(threshold=0.5),
    # ]

    # test_epoch = smp.utils.train.ValidEpoch(
    #     model=best_model,
    #     loss= loss,
    #     metrics=metrics,
    #     device=DEVICE,
    # )

    # logs = test_epoch.run(test_dataloader)

    # test dataset without transformations for image visualization
    test_dataset_vis = Dataset(
        TEST_IMG_DIR, TEST_MASK_DIR, 
        classes=CLASSES,
    )

    #save prediction masks
    print("\nMaking predictions:")
    loop = tqdm(test_dataset)
    for idx, (image, gt_mask) in enumerate(loop):
        print(image.shape)
        print(gt_mask.shape)
        gt_mask = gt_mask.squeeze()
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        # pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        torchvision.utils.save_image(pr_mask, f"{OUTPUT_DIR}/pred_{idx + 1}.png")

if __name__ == "__main__":
    main()