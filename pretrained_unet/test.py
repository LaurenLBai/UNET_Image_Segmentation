import torch
from utils import load_checkpoint

TEST_IMG_DIR = "../Datasets/shoreline_ready_data/val/images/"
TEST_MASK_DIR = "../Datasets/shoreline_ready_data/val/masks/"

# load_checkpoint(torch.load("best_model.pth"), model)
best_model = torch.load('./best_model.pth')

# create test dataset
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)