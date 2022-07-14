import os
import cv2
from PIL import Image
import numpy as np
from patchify import patchify


patch_size = 256

img_dir = "../Datasets/Shoreline_Dataset/images/"
images = os.listdir(img_dir)
id = 0

for i, image_name in enumerate(images):
    image = cv2.imread(img_dir + image_name, 1)
    SIZE_X = (image.shape[1]//patch_size)*patch_size
    SIZE_Y = (image.shape[0]//patch_size)*patch_size
    image = Image.fromarray(image)
    image = image.crop((0,0, SIZE_X, SIZE_Y))
    image = np.array(image)

    print("Now patchifying image:", img_dir + image_name)
    patches_img = patchify(image, (256, 256, 3), step=256)


    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            id += 1
            single_patch_img = patches_img[i, j, :, :]
            single_patch_img = single_patch_img[0]
            cv2.imwrite("256_Water/images/" + "IMG_" + str(id) + ".png", single_patch_img)


mask_dir = "../Datasets/Shoreline_Dataset/masks/"
masks = os.listdir(mask_dir)
id = 0


for i, mask_name in enumerate(masks):
    mask = cv2.imread(mask_dir + mask_name, 0)
    SIZE_X = (mask.shape[1]//patch_size)*patch_size
    SIZE_Y = (mask.shape[0]//patch_size)*patch_size
    mask = Image.fromarray(mask)
    mask = mask.crop((0,0, SIZE_X, SIZE_Y))
    mask = np.array(mask)

    print("Now patchifying mask:", mask_dir + mask_name)
    patches_mask = patchify(mask, (256, 256), step=256)

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            id += 1
            single_patch_mask = patches_mask[i, j, :, :]
            cv2.imwrite("256_Water/masks/" + "IMG_" + str(id) + ".png", single_patch_mask)

