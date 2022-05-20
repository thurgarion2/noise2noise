# AUTOGENERATED! DO NOT EDIT! File to edit: 00_training.ipynb (unless otherwise specified).

__all__ = ['ToFloatImage', 'Noise2NoiseDataset', 'psnr_metric', 'train', 'eval_model', 'row', 'col', 'show_results',
           'show_worst', 'plot_psnr_distribution', 'plot_psnr', 'plot_losses', 'lr_finder']

# Cell
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .helpers import *

# Cell
class ToFloatImage:

    def __init__(self):
       pass

    def __call__(self, img):
        return to_float_image(img)

# Cell
class Noise2NoiseDataset(Dataset):
    def __init__(self, img_path, transform=None, target_transform=None, input_target_transforms = None, nb_samples=None):

        self.imgs, self.targets = load_images(img_path)
        self.transform = transform
        self.target_transform = target_transform
        self.input_target_transforms = input_target_transforms
        self.nb_samples = nb_samples


    def __len__(self):
        return  self.nb_samples if self.nb_samples else len(self.imgs)

    def __getitem__(self, idx):
        image, target = self.imgs[idx], self.targets[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        if self.input_target_transforms:
            image, target = self.input_target_transforms((image,target))

        return image, target

# Cell
def psnr_metric():
    return lambda denoised , ground_truth: psnr(denoised.clip(0,1) , ground_truth)

# Cell
def train(model,
          train_loader,
          test_loader,
          optim,
          criterion,
          metric,
          gradient_steps=1000,
          samples_steps=100,
          lr_scheduler=None,
          lr_steps=1):
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

        if (steps+1)%samples_steps==0:

            if train_loader:
              train_loss.append((steps, eval_model(model, train_loader, criterion, max_iter=5)))
            if test_loader:
              test_loss.append((steps, eval_model(model, test_loader, criterion)))
              test_metric.append((steps,eval_model(model, test_loader, metric)))

        if lr_scheduler and (steps+1)%lr_steps==0:
          lr_scheduler.step()


    return train_loss, test_loss, test_metric




def eval_model(model, loader, metric, max_iter=None):
    losses = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    with torch.no_grad():
        for i, (input, target) in enumerate(iter(loader)):

            input, target = input.to(device), target.to(device)
            output = model(input)
            losses.append(metric(output, target).item())

            if max_iter and i>=max_iter:
                break


    return sum(losses)/len(losses)

# Cell
#hide
def row(i, nb_cols):
    return i//nb_cols
def col(i, nb_cols):
    return (i%nb_cols)*2


# Cell

def show_results(model, dataloader, nb_cols=3, nb_rows=2):
    _, axis = plt.subplots(nb_rows,nb_cols*2, figsize=(nb_cols*6,3*nb_cols))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input, target = iter(dataloader).next()
    model.to("cpu")

    with torch.no_grad():
        output = model(input)
        nb_examples = nb_cols*nb_rows



        for i in range(min(nb_examples, len(output))):
            r, c = row(i,nb_cols), col(i, nb_cols)
            axis[r,c].set_title(f'psnr :{psnr(output[i] , target[i]).item():.{4}f}')
            show_img(output[i], axis=axis[r,c])
            show_img(target[i], axis=axis[r,c+1])
    model.to(device)




# Cell

def show_worst(model, dataloader, nb_cols=3, nb_rows=2):
    _, axis = plt.subplots(nb_rows,nb_cols*2, figsize=(nb_cols*6,3*nb_cols))
    nb_examples = nb_cols*nb_rows
    psnrs = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    with torch.no_grad():
        for input, target in iter(dataloader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            for i, p in enumerate(psnr(output , target, dim=[1,2,3])):
                psnrs.append((p,(output[i],target[i])))


    psnrs.sort(key=lambda x:x[0])
    for i in range(min(nb_examples, len(output))):
            r, c = row(i,nb_cols), col(i, nb_cols)
            p, (output,target) = psnrs[i][0], psnrs[i][1]
            axis[r,c].set_title(f'psnr :{p:.{4}f}')
            show_img(output.to("cpu"), axis=axis[r,c])
            show_img(target.to("cpu"), axis=axis[r,c+1])

# Cell

def plot_psnr_distribution(model, dataloader):
    psnrs = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    with torch.no_grad():
        for input, target in iter(dataloader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            psnrs.extend(psnr(output , target, dim=[1,2,3]).tolist())
    plt.hist(psnrs, bins=30)
    plt.xlabel('psnr')
    plt.ylabel('number images')


# Cell
def plot_psnr(psnrs, labels, axis):
    for psnr, label in zip(psnrs,labels):
        steps, psnr = zip(*psnr)
        axis.plot(steps, psnr, label=label)


    axis.set_xlabel('gradient steps')
    axis.legend()
    axis.set_ylabel('psnr')

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

# Cell

def lr_finder(model, loader, optim_class, criterion, gradient_steps=1000, samples_steps=100, lr_steps=10):
  lr_steps = gradient_steps//lr_steps
  lr = 1e-8

  optim = optim_class(model.parameters(), lr=lr)
  lrs = torch.logspace(-8,0,lr_steps)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: lrs[x]/1e-8)

  losses, _, _ = train(model, loader, None, optim, criterion, None,
          gradient_steps=gradient_steps,
          samples_steps=samples_steps,
          lr_scheduler=scheduler,
          lr_steps=10)


  _, losses = zip(*losses)
  idxs = torch.logspace(-8,0,gradient_steps//samples_steps)[1:]
  plt.plot(idxs,list(losses))
  plt.xscale('log')
  plt.xlabel('lr rate')
  plt.ylabel('loss')
