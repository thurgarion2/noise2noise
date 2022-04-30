# AUTOGENERATED! DO NOT EDIT! File to edit: 00_training.ipynb (unless otherwise specified).

__all__ = ['ToFloatImage', 'Noise2NoiseDataset', 'psnr_metric', 'train', 'eval_model', 'show_results', 'plot_losses']

# Cell
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .helpers import *
from .unet import *

# Cell
class ToFloatImage:

    def __init__(self):
       pass

    def __call__(self, img):
        return to_float_image(img)

# Cell
class Noise2NoiseDataset(Dataset):
    def __init__(self, img_path, transform=None, target_transform=None):

        self.imgs, self.targets = load_images(img_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image, target = self.imgs[idx], self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

# Cell
def psnr_metric():
    return lambda denoised , ground_truth: psnr(denoised.clip(0,1) , ground_truth)

# Cell
def train(model, train_loader, test_loader, optim, criterion, metric, gradient_steps=1000, samples_steps=100):
    iterator = iter(train_loader)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss = []
    test_loss = []
    test_metric = []


    for steps in tqdm(range(gradient_steps)):
        input, target = None, None
        try:
            input, target = iterator.next()
        except StopIteration:
            iterator = iter(train_loader)
            input, target = iterator.next()
        input, target = input.to(device), target.to(device)
        output = model(input)
        loss = criterion(output, target)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if steps%samples_steps==0 and steps>0:
            train_loss.append((steps, eval_model(model, train_loader, criterion)))
            test_loss.append((steps, eval_model(model, test_loader, criterion)))
            test_metric.append((steps,eval_model(model, test_loader, metric)))

    return train_loss, test_loss, test_metric




def eval_model(model, loader, metric):
    losses = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iterator = iter(loader)

    with torch.no_grad():
        for loop in range(5):
            input, target = iterator.next()
            input, target = input.to(device), target.to(device)

            output = model(input)
            losses.append(metric(output, target).item())


    return sum(losses)/len(losses)

# Cell

def show_results(model, dataloader, nb_examples=4):
    _, axis = plt.subplots(4,2, figsize=(8,4*nb_examples))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input, target = iter(dataloader).next()
    model.to("cpu")

    with torch.no_grad():
        output = model(input)

        for i in range(min(nb_examples, len(output))):
            show_img(output[i], axis=axis[i,0])
            show_img(target[i], axis=axis[i,1])
            print(f'psnr :{psnr(output[i] , target[i]).item()}')
    model.to(device)




# Cell
def plot_losses(train_losses, test_losses, test_metric):
    _, (loss_axis, metric_axis) = plt.subplots(1,2, figsize=(16,4))
    steps, train_losses = zip(*train_losses)
    _, test_losses = zip(*test_losses)
    _, test_metric = zip(*test_metric)

    loss_axis.plot(steps, train_losses,color='black',label='train')
    loss_axis.plot(steps, test_losses,color='grey',label='test')
    loss_axis.set_ylabel('mse')
    loss_axis.set_xlabel('gradient steps')
    loss_axis.legend()


    metric_axis.plot(steps, test_metric,color='grey', label='test')
    metric_axis.set_xlabel('gradient steps')
    metric_axis.legend()
    metric_axis.set_ylabel('psnr')