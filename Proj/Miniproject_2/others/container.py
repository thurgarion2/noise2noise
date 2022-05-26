import torch

from .module import Module

class Sequential(Module):
    ''' Container like torch.nn.Sequential to put together an
    arbitrary configuration of modules together. '''
    def __init__(self, *modules):
        '''Sequential constructor

        :modules: Modules configuration to put together
        '''
        self.modules = list(modules)

    def forward(self, input_: torch.Tensor):
        '''Sequential forward pass
        
        :input_: Previous layer output

        :returns: Forward pass through underlying modules output
        '''
        # Initial input
        x = input_.clone()
        # Sequentially go through the underlying modules
        for mod in self.modules:
            # Pass output of the previous module to the next one as input
            x = mod.forward(x)

        # Return the output from the last underlying module
        return x

    def backward(self, d_out: torch.Tensor):
        '''Sequential backward pass
        
        :d_out: Gradient from the next layer

        :returns: Backward pass through underlying modules output
        '''
        # Initial input
        x = d_out.clone()
        # Sequentially go through underlying modules in reverse order
        for mod in self.modules[::-1]:
            # Propagate gradient from next layer to previous layer
            x = mod.backward(x)
          

        # Return gradient propagated through underlying modules    
        return x
        

    def param(self):
        '''Retrive all parameters from the underlying modules
        
        :returns: list of pairs (parameter, gradient) of tensors of the same size
        '''
        # Initialize empty parameter list
        param_list = []
        # Retrieve from each underlying module
        for mod in self.modules:
            for p in mod.param():
                param_list.append(p)
        
        return param_list