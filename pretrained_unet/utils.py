from matplotlib import image
import torch
import torchvision
from dataset import Dataset
from torch.utils.data import DataLoader
import albumentations as albu
import matplotlib.pyplot as plt
import random
import os
import cv2

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_augmentation,
    val_augmentation,
    num_workers=4,
    preprocessing_fn = None, 
    classes = None,
):
    train_ds = Dataset(
        images_dir=train_dir,
        masks_dir=train_maskdir,
        classes = classes,
        augmentation=train_augmentation,
        preprocessing= get_preprocessing(preprocessing_fn), 
        

    )   

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    val_ds = Dataset(
        images_dir=val_dir,
        masks_dir=val_maskdir,
        classes = classes,
        augmentation=val_augmentation,
        preprocessing= get_preprocessing(preprocessing_fn),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, val_loader

def print_examples(img_dir, mask_dir):
    img_list = os.listdir(img_dir)
    mask_list = os.listdir(mask_dir)
    num_images = len(os.listdir(img_dir))
    img_num = random.randint(0, num_images-1)
    img_for_plot = cv2.imread(img_dir + img_list[img_num], 1)
    print(img_for_plot.shape)
    img_for_plot = cv2.cvtColor(img_for_plot, cv2.COLOR_BGR2RGB)
    mask_for_plot = cv2.imread(mask_dir + mask_list[img_num], 0)
    print(mask_for_plot.shape)

    plt.figure(figsize=(12,8))
    plt.subplot(121)
    plt.imshow(img_for_plot)
    plt.title('Image')
    plt.subplot(122)
    plt.imshow(mask_for_plot, cmap = 'gray')
    plt.title('Mask')
    plt.show()



def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
