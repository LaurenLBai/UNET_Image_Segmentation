import torch
from utils import (
    get_preprocessing,
    visualize,
    
)
from dataset import Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np

TEST_IMG_DIR = "../Datasets/shoreline_ready_data/test/images/"
TEST_MASK_DIR = "../Datasets/shoreline_ready_data/test/masks/"
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['water']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def main():

    #load the model
    best_model = torch.load('best_model.pth')
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold = 0.5),
    ]

    # create test dataset
    test_dataset = Dataset(
        TEST_IMG_DIR, 
        TEST_MASK_DIR, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    #create dataloader
    test_dataloader = DataLoader(test_dataset)
    #create test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )
    logs = test_epoch.run(test_dataloader)
    test_dataset_vis = Dataset(
        TEST_IMG_DIR, 
        TEST_MASK_DIR, 
        classes=CLASSES,
        )
    for i in range(5):
        n = np.random.choice(len(test_dataset))
        
        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
        visualize(
            image=image_vis, 
            ground_truth_mask=gt_mask, 
            predicted_mask=pr_mask
        )

if __name__ == "__main__":
    main()