import torch
from torch import empty

from module import Module

class Linear(Module):
    '''Linear layer'''
    def __init__(self, in_features, out_features):
        '''Linear constructor
        
        :in_features: size of each input sample

        :out_features: size of each output sample
        '''
        # Initialize weights and bias
        sqrt_k = (1.0 / in_features) ** .5
        self.W = empty((out_features, in_features)).uniform_(-sqrt_k, sqrt_k)
        self.b = empty((out_features, 1)).uniform_(-sqrt_k, sqrt_k)

        # Initialize weights & bias gradients
        self.dW = self.W.new_zeros(self.W.size())
        self.db = self.b.new_zeros(self.b.size())

    def forward(self, input_: torch.Tensor):
        '''Applies linear transformation to incoming data
        
        :input_: Input tensor from previous layer output

        :returns: Linear transformation w.r.t weights and bias
        '''
        # Save x for backwards pass
        self.input_ = input_.clone()

        # Apply transformation
        return self.W @ input_ + self.b
        

    def backward(self, d_out): # TODO check correctness
        '''Linear backward pass
        
        :d_out: Gradient from next layer

        :returns: Loss gradient w.r.t input
        '''
        # Update weight and bias gradient
        self.dW = d_out.mm(self.input_.T)
        self.db = d_out.clone()
        
        return self.W.T.mm(d_out)

    def param(self):
        '''Return weight and bias parameters'''
        return [(self.W, self.dW), (self.b, self.db)]
