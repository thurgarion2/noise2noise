import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

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

# TODO: Modules Conv2d, TransposeConv2d, NearestUpsampling, ReLU, Sigmoid, MSE, SGD, Sequential