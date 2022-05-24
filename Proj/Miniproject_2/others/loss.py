import torch

from .module import Module

class MSE(Module):
    '''Mean squared error loss'''
    def forward(self, input_: torch.Tensor, target: torch.Tensor):
        '''Performs MSE forward pass
        
        :input_: Output from previous layer

        :target: True labels

        :returns: Mean squared error w.r.t. input and target labels
        '''
        # Compute error
        error = target.sub(input_)
        # Save error tensor for backward pass
        self.error = error.clone()
        # Compute MSE
        return error.pow(2).mean()

    def backward(self):
        '''Performs MSE backwards pass
        
        :returns: MSE gradient
        '''
        return self.error.mean().mul(-2)

    def param(self):
        '''MSE is a parameterless module'''
        return []