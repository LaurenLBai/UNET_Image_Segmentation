# UNET_Image_Segmentation

The UNET machine learning model is used for image segmentation in order to classify the boudraries of certain classes. The model and its initial weights comes from the Segmentation Models library. The model is trained on input images 256 x 256 so they must be split into patches which can be found in the code. The Smoothly Blended Patches library is used to blend images back together from these patches for the final output
