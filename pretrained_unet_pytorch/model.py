import segmentation_models_pytorch as smp
from torch import classes

model = smp.Unet(
    encoder_name = "resnet34",
    encoder_weights= "imagenet",
    in_channels= 3,
    classes = len(classes)
)
print(model)

