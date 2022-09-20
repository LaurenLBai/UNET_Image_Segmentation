# https://youtu.be/0W6MKZqSke8
"""
Author: Dr. Sreenivas Bhattiprolu 
Prediction using smooth tiling as descibed here...
https://github.com/Vooban/Smoothly-Blend-Image-Patches
"""

import cv2
import torch
import numpy as np

from matplotlib import pyplot as plt
import segmentation_models_pytorch as smp

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from smooth_tiled_predictions import predict_img_with_smooth_windowing


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
preprocess_input = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


img = cv2.imread("../Datasets/Shoreline_Test_Dataset/images/p50.JPG")  
input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
input_img = preprocess_input(input_img)

original_mask = cv2.imread("../Datasets/Shoreline_Test_Dataset/masks/p50.png")
original_mask = original_mask[:,:,0]  #Use only single channel...
#original_mask = to_categorical(original_mask, num_classes=n_classes)


# load best saved checkpoint
model = torch.load('best_model.pth')
model.to(DEVICE).eval()
                  
# size of patches
patch_size = 256

# Number of classes 
n_classes = 1

         
###################################################################################
#Predict using smooth blending

# Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
# Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
predictions_smooth = predict_img_with_smooth_windowing(
    input_img,
    window_size=patch_size,
    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
    nb_classes=n_classes,
    pred_func=(
        lambda img_batch_subdiv: model((img_batch_subdiv ))
    )
)


final_prediction = np.argmax(predictions_smooth, axis=2)

#Save prediction and original mask for comparison
plt.imsave( "../Datasets/Shoreline_Test_Dataset/predictions/pred_1.png", final_prediction)
plt.imsave("../Datasets/Shoreline_Test_Dataset/masks/og.png", original_mask)
###################


plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title('Testing Image')
plt.imshow(img)
plt.subplot(222)
plt.title('Testing Label')
plt.imshow(original_mask)
plt.subplot(223)
plt.title('Prediction with smooth blending')
plt.imshow(final_prediction)
plt.show()

#############################