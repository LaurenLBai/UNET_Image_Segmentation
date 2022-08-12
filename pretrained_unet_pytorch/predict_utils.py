
from predict_dataset import Dataset
from torch.utils.data import DataLoader
import albumentations as albu


def to_tensor(x, **kwargs):
    x = x.transpose(2, 0, 1).astype('float32')
    return x


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
    subdivs,
    batch_size,
    num_workers=4,
    classes = None,
    preprocessing_fn = None, 
):
    predict_ds = Dataset(
        subdivs,
        preprocessing= get_preprocessing(preprocessing_fn), 
    )   

    predict_loader = DataLoader(
        predict_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    return predict_loader