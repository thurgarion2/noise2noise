import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
import math

from .module import Module

class Conv2d(Module):
    '''Conv2d module implemented by a linear function'''
    def __init__(self, in_channels=3, out_channels=3, kernel_size=(2, 2), stride=1):
        '''Conv2d module constructor
        
        :in_channels: (int) Number of channels in the input image, default = 3

        :out_channels: (int) Number of channels produced by the convolution, default = 3

        :kernel_size: (tuple) Size of the convolving kernel, default = (2, 2)

        :stride: (int) Stride of the convolution, default = 1

        :padding: (int) controls the amount of padding applied to the input
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        

        # Initialize weights & bias
        sqrt_k = 1 / (in_channels * kernel_size[0] * kernel_size[1]) ** 0.5

        self.weight = empty((out_channels, in_channels, kernel_size[0], kernel_size[1])).uniform_(-sqrt_k, sqrt_k)
        self.bias = empty((out_channels)).uniform_(-sqrt_k, sqrt_k)

        # Initialize weights & bias gradients
        self.dW = self.weight.new_zeros(self.weight.size())
        self.db = self.bias.new_zeros(self.bias.size())

    def forward(self, input_):
        '''Conv2d forward pass
        
        :input_: (torch.Tensor) Input tensor to apply convolution on, with size (N, C_in, H, W)

        :returns: 2D convolution applied on input signal, with size (N, C_out, H_out, W_out)
        '''
        # Clone the input for the gradients

        self.input_shape = input_.shape[2:]
        input_unfolded = unfold(input_, kernel_size=self.kernel_size, stride=self.stride)
       
        self.input_ = input_unfolded.clone()
        input_convolved = self.weight.view(self.out_channels, -1) @ input_unfolded + self.bias.view(1, -1, 1)
        return input_convolved.view(
            -1, # |B|
            self.out_channels, # C_out
            math.floor((input_.shape[2] - self.kernel_size[0])/self.stride) + 1, # H_out
            math.floor((input_.shape[3] - self.kernel_size[1])/self.stride) + 1  # W_out
        )
    def __call__(self, input_):
        return self.forward(input_)


    def backward(self, d_out):
        '''Conv2d backward pass
        
        :d_out: (tensor) Gradient w.r.t output

        :returns: Propagated loss gradient
        '''
        d_out = d_out.view(d_out.size(0),self.out_channels,-1)
     
        # Update bias gradient
        self.db += d_out.sum([0, 2])

        # Update weight gradient        
        dW = d_out @ self.input_.transpose(-1,-2)
        self.dW += dW.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        # Propagate loss gradient
        out_ = self.weight.view(self.out_channels, -1).T@d_out
        print(out_)
        return fold(out_,output_size=self.input_shape, kernel_size=self.kernel_size, stride=self.stride)

    def param(self):
        '''Return Conv2d weight and bias parameters'''
        return [(self.weight, self.dW), (self.bias, self.db)]

class TransposeConv2d(Module): # TODO
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding=0
        self.dilation=1

        # Initialize weights & bias
        sqrt_k = 1 / (out_channels * kernel_size[0] * kernel_size[1]) ** 0.5

        self.weight = empty((in_channels, out_channels, kernel_size[0], kernel_size[1])).uniform_(-sqrt_k, sqrt_k)
        self.bias = empty((out_channels)).uniform_(-sqrt_k, sqrt_k)

        # Initialize weights & bias gradients
        self.dW = self.weight.new_zeros(self.weight.size())
        self.db = self.bias.new_zeros(self.bias.size())

    def forward(self, input_):
        '''TransposeConv2d forward pass
        
        :input_: (tensor) input tensor

        :returns: 2D transposed convolution operator applied over input
        '''
        pass

    def backward(self, d_out):
        '''TransposeConv2d backward pass
        
        :d_out: (tensor) Gradient w.r.t. output (backpropagated gradient from next layers)

        :returns: Loss gradient to propagate
        '''
        pass

    def param(self):
        '''Return TransposeConv2d weight and bias parameters'''
        return [(self.weight, self.dW), (self.bias, self.db)]

class Upsampling(Module): # TODO (underlying is transposeConv2d)
    '''Upsampling module'''
    def __init__(self):
        '''Upsampling constructor'''
        pass