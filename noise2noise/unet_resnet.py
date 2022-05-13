# AUTOGENERATED! DO NOT EDIT! File to edit: 00_unet_resnet.ipynb (unless otherwise specified).

__all__ = ['ResNetBlock', 'ResizeBlock', 'ResNetUnet', 'ResNetUnetNotPretrained']

# Cell
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from .helpers import *

# Cell
#maybe implement basic block

class ResNetBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(channels))

    def forward(self, x):
        return F.relu(self.layers(x)+x)

class ResizeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, size):
        super().__init__()
        self.resize_channel = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.size = size

    def forward(self, x):
        return self.resize_channel(F.interpolate(x, size=self.size))


# Cell

class ResNetUnet(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained = True)
        resnet_layers = list(resnet.children())
        self.encoder = nn.ModuleList([nn.Sequential(*resnet_layers[:4]),*resnet_layers[4:-2]])


        self.decoder_layers = []
        x = torch.zeros((2, 3, 32, 32))
        for i, l in enumerate(self.encoder):
            x_next = l(x)
            in_channels = x_next.size(1)
            out_channels = x.size(1)
            size = (x.size(2),x.size(3))

            if i>0:
              self.decoder_layers.append(nn.Sequential(ResNetBlock(in_channels),
                                                     ResizeBlock(in_channels, out_channels, size)))

            x = x_next
        self.decoder_layers.reverse()
        self.decoder = nn.ModuleList(self.decoder_layers)
        self.middle = ResNetBlock(in_channels)
        self.to_32 = nn.Sequential(ResizeBlock(64,32,(32,32)), ResNetBlock(32))
        self.to_3 = nn.Conv2d((32+3), 3, kernel_size=3, padding=1)

    def forward(self, x):
        img = x
        intermediary_x = []
        for l in self.encoder:
            x = l(x)
            intermediary_x.append(x)


        intermediary_x.reverse()
        x = self.middle(x)

        for l, x_other in zip(self.decoder,intermediary_x):
            x = l(x+x_other)

        x = self.to_32(x+intermediary_x[-1])
        return self.to_3(torch.cat([x,img],dim=1))

# Cell

class ResNetUnetNotPretrained(nn.Module):
  def __init__(self):
    super().__init__()