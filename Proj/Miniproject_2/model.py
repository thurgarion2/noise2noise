import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

# We turn autograd globally off
torch.set_grad_enabled(False)

class Module(object):
    '''Suggested simple structure for implemented modules to inherit.
    Some modules may require additional methods, and some modules may keep track 
    of information from the forward pass to be used in the backward.'''
    def forward(self, *input):
        '''Should get for input and returns, a tensor or a tuple of tensors.'''
        raise NotImplementedError
    def backward(self, *gradwrtoutput):
        '''Should get as input a tensor or a tuple of tensors containing the gradient 
        of the loss with respect to the module's output, accumulate the gradient w.r.t.
        the parameters, and return a tensor or a tuple of tensors containing the gradient
        of the loss wrt the module's input.'''
        raise NotImplementedError
    def param(self):
        '''Should return a list of pairs composed of a parameter tensor and a gradient 
        tensor of the same size. This list should be empty for parameterless modules (such as ReLU).'''
        return []
    
class Model():
    def __init__(self) -> None:
        '''Instantiate model + optimizer + loss function + any other stuff needed'''
        pass

    def load_pretrained_model(self) -> None:
        '''This loads the parameters saved in bestmodel.pth into the model'''
        pass

    def train(self, train_input, train_target, num_epochs) -> None:
        '''
        :train_input: tensor of size (N, C, H, W) containing a noisy version of the images

        :train_target: tensor of size (N, C, H, W) containing another noisy version of the same
        images, which only differs from the input by their noise

        :num_epochs: number of training epochs
        '''
        pass

    def predict(self, test_input) -> torch.Tensor:
        '''
        :test_input: tensor of size (N1, C, H, W) with values in range 0-255 that has to
        be denoised by the trained or the loaded network.

        :returns a tensor of size (N1, C, H, W) with values in range 0-255
        '''
        pass

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
        return input_.maximum(input_.new_zeros(input_.size()))

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
        return self.error.mean().mul(2)

    def param(self):
        '''MSE is a parameterless module'''
        return []
        
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


# TODO: Modules Conv2d, TransposeConv2d, NearestUpsampling, SGD