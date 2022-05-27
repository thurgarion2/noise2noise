# AUTOGENERATED! DO NOT EDIT! File to edit: 00_training_pytorch.ipynb (unless otherwise specified).

__all__ = ['to_float_image', 'ToFloatImage']

# Cell
import torch

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def to_normalize(img, mean,std):
    return (img-mean)/std

class Compose:
    
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ToNormalize:

    def __init__(self, mean,std):
        self.mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        self.std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

    def __call__(self, img):
        return to_normalize(img,self.mean,self.std)

# Cell
def to_float_image(image):
    return image.float()/255

# Cell
class ToFloatImage:

    def __init__(self):
       pass

    def __call__(self, img):
        return to_float_image(img)