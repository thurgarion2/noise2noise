import torch

from module import Module

class ReLU(Module):
    '''ReLU activation'''
    def forward(self, input_: torch.Tensor):
        '''Performs ReLU forward pass
        
        :input: Input tensor from previous layer output

        :returns: ReLU activation applied on input
        '''
        # Save input for backward pass
        self.input_ = input_.clone()
        # Apply ReLU = max(0, x)
        return input_.max(input_.new_zeros(input_.size()))

    def backward(self, d_out: torch.Tensor):
        '''Performs ReLU backward pass
        
        :d_out: Derivatives from the next layer

        :returns: Propagated gradient
        '''
        return d_out * (self.input_ > 0)

    def param(self):
        '''ReLU is a parameterless module'''
        return []

class Sigmoid(Module):
    '''Sigmoid activation'''
    def forward(self, input_: torch.Tensor):
        '''Performs Sigmoid forward pass
        
        :input_: Input tensor from previous layer output

        :returns: Sigmoid activation applied on input
        '''
        # Apply sigmoid
        sigma = input_.sigmoid()
        # Save for backward pass
        self.sigma = sigma.clone()
        return sigma
    
    def backward(self, d_out: torch.Tensor):
        '''Performs Sigmoid backward pass
        
        :d_out: Derivatives from the next layer

        :returns: Propagated gradient
        '''
        return d_out * (self.sigma * (1 - self.sigma))
    
    def param(self):
        '''Sigmoid is a parameterless module'''
        return []