import os
import cv2
from torch.utils.data import Dataset as BaseDataset
import numpy as np

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['water']
    
    def __init__(
            self, 
            subdivs,
            preprocessing=None,
    ):
        self.subdivs = subdivs
        self.preprocessing = preprocessing
        self.ids = subdivs.shape[0]
    
    def __getitem__(self, i):
        
        # read data
        image = self.subdivs[i]
        print(image.shape)
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
            
        return image
        
    def __len__(self):
        return self.ids