import torch
import segmentation_models_pytorch as smp
import albumentations as A
from dataset import Dataset


from utils import (
    load_checkpoint,
    get_loaders,
    save_predictions_as_imgs,
    print_examples,
    visualize,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 1
NUM_WORKERS = 4
IMAGE_HEIGHT = 256  
IMAGE_WIDTH = 256  
LOAD_MODEL = False
TRAIN_IMG_DIR = "../Datasets/shoreline_ready_data/train/images/"
TRAIN_MASK_DIR = "../Datasets/shoreline_ready_data/train/masks/"
VAL_IMG_DIR = "../Datasets/shoreline_ready_data/val/images/"
VAL_MASK_DIR = "../Datasets/shoreline_ready_data/val/images/"

def main():
   
    #print examples
    for i in range(0,3):
        print_examples(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    
    #print examples
    # cont = 'c'
    # i = 0
    # dataset = Dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, classes=['water'])
   
    # while(cont == 'c'):
    #     image, mask = dataset[i] # get some sample
    #     visualize(
    #         image=image, 
    #         mask=mask.squeeze(),
    #     )
    #     i+=1
    #     cont = input("enter c to continue: ")
    #create model
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['water']
    ACTIVATION = 'sigmoid' #softmax for multiclass

    model = smp.Unet(
        encoder_name = ENCODER,
        encoder_weights= ENCODER_WEIGHTS,
        classes = len(CLASSES),
        activation = ACTIVATION,
    )


    preprocessign_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    #create augmentations
    train_augmentation = A.Compose(
        [
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
        ],
    )

    val_augmentation = A.Compose(
        [
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
        ],
    )

    #create loaders
    train_loader, train_ds, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_augmentation,
        val_augmentation,
        NUM_WORKERS,
        preprocessign_fn,
        classes = CLASSES, 
    )
    
    
    #define loss and optimizer
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold = 0.5),
        smp.utils.metrics.Accuracy(threshold = 0.5),

    ]
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr = LEARNING_RATE),
    ])
   
    #load the model if set
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    #create epoch runners, iterate through dataloader's samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss= loss,
        metrics = metrics,
        optimizer = optimizer,
        device = DEVICE,
        verbose = True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss = loss,
        metrics = metrics,
        device = DEVICE,
        verbose = True,
    )
    
    # train model
    max_score = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch: {epoch+1}")
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

        if epoch == 20:
            optimizer.param_groups[0]['lr'] = 1e-5
            print("Decreasing learning rate to 1e-5")

        
        # #print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder="unet/saved_images/", device=DEVICE
        # )
        #get a summary

if __name__ == "__main__":
    main()