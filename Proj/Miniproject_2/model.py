import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
import math

from pathlib import Path

# We turn autograd globally off
torch.set_grad_enabled(False)
# Control randomness for reproducibility
torch.manual_seed(42)

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

class Conv2d(Module): # TODO
    '''Conv2d module implemented by a linear function'''
    def __init__(self, in_channels=3, out_channels=3, kernel_size=(2, 2), stride=1):
        '''Conv2d module constructor
        
        :in_channels: (int) Number of channels in the input image, default = 3

        :out_channels: (int) Number of channels produced by the convolution, default = 3

        :kernel_size: (tuple) Size of the convolving kernel, default = (2, 2)

        :stride: (int) Stride of the convolution, default = 1
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Initialize weights & bias
        sqrt_k = 1 / (in_channels * kernel_size[0] * kernel_size[1]) ** 0.5

        self.W = empty((out_channels, in_channels, kernel_size[0], kernel_size[1])).uniform_(-sqrt_k, sqrt_k)
        self.b = empty((out_channels)).uniform_(-sqrt_k, sqrt_k)

        # Initialize weights & bias gradients
        self.dW = self.W.new_zeros(self.W.size())
        self.db = self.b.new_zeros(self.b.size())

    def forward(self, input_):
        '''Conv2d forward pass
        
        :input_: (torch.Tensor) Input tensor to apply convolution on, with size (N, C_in, H, W)

        :returns: 2D convolution applied on input signal, with size (N, C_out, H_out, W_out)
        '''
        # Clone the input for the gradients
        self.input_ = input_.clone()

        input_unfolded = unfold(input_, kernel_size=self.kernel_size, stride=self.stride)
        input_convolved = self.W.view(self.out_channels, -1) @ input_unfolded + self.b.view(1, -1, 1)
        return input_convolved.view(
            -1, # |B|
            self.out_channels, # C_out
            math.floor((input_.shape[2] - self.kernel_size[0])/self.stride) + 1, # H_out
            math.floor((input_.shape[3] - self.kernel_size[1])/self.stride)+ 1   # W_out
        )


    def backward(self, d_out):
        '''Conv2d backward pass
        
        :d_out: (tensor) Gradient w.r.t output

        :returns: Propagated loss gradient
        '''
        # Update bias gradient
        self.db += d_out.sum([0, 2, 3])
        # Update weight gradient
        self.dW += NotImplementedError
        # Propagate loss gradient
        return NotImplementedError


    def param(self):
        '''Return Conv2d weight and bias parameters'''
        return [(self.W, self.dW), (self.b, self.db)]

class TransposeConv2d(Module): # TODO
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Initialize weights & bias
        sqrt_k = 1 / (in_channels * kernel_size[0] * kernel_size[1]) ** 0.5

        self.W = empty((out_channels, in_channels, kernel_size[0], kernel_size[1])).uniform_(-sqrt_k, sqrt_k)
        self.b = empty((out_channels)).uniform_(-sqrt_k, sqrt_k)

        # Initialize weights & bias gradients
        self.dW = self.W.new_zeros(self.W.size())
        self.db = self.b.new_zeros(self.b.size())

    def forward(self):
        pass

    def backward(self):
        pass

    def param(self):
        '''Return TransposeConv2d weight and bias parameters'''
        return [(self.W, self.dW), (self.b, self.db)]

class Upsampling(Module): # TODO (underlying is transposeConv2d)
    pass

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

class SGD:
    '''Stochastic gradient descent optimizer'''
    def __init__(self, params, learning_rate=0.05):
        '''SGD optimizer constructor
        
        :params: (iterable) - iterable of parameters to optimize

        :learning_rate: (float) - learning rate, default = 0.05
        '''
        self.params = params
        self.lr = learning_rate

    def zero_grad(self):
        '''Sets the gradients of all optimized torch.Tensor to 0'''
        for _, g in self.params:
            g.zero_()

    def step(self):
        '''Performs single optimization step'''
        for p, g in self.params:
            p -= self.lr * g

class Model():
    '''Model'''
    def __init__(self) -> None:
        '''Instantiate model + optimizer + loss function + any other stuff needed'''
        # Instantiate model
        self.model = Sequential(
            Conv2d(stride=2),
            ReLU,
            Conv2d(stride=2),
            ReLU,
            Upsampling,
            ReLU,
            Upsampling,
            Sigmoid
        )
        # Instantiate optimizer
        self.optimizer = SGD(self.model.param())
        # Choose loss function
        self.criterion = MSE

        # Default mini batch size
        self.batch_size = 100

        # Default path to save model
        self.model_path = Path(__file__).parent / "bestmodel.pth"

    def load_pretrained_model(self) -> None:
        '''This loads the parameters saved in bestmodel.pth into the model'''
        self.model = torch.load(self.model_path)
        self.optimizer = SGD(self.model.param())

    def train(self, train_input: torch.Tensor, train_target: torch.Tensor, 
        num_epochs=50, verbose=False) -> None:
        ''' Train model

        :train_input: tensor of size (N, C, H, W) containing a noisy version of the images

        :train_target: tensor of size (N, C, H, W) containing another noisy version of the same
        images, which only differs from the input by their noise

        :num_epochs: (int) number of training epochs, default = 50

        :verbose: (bool) Choice of verbose execution, default = False
        '''
        for e in range(num_epochs):
            epoch_loss = 0.0
            for inputs, targets in zip(train_input.split(self.batch_size), 
                                       train_target.split(self.batch_size)):
                output = self.model.forward(inputs)
                epoch_loss += self.criterion.forward(output, targets)
                self.optimizer.zero_grad()
                d_loss = self.criterion.backward()
                self.model.backward(d_loss)
                self.optimizer.step()

            if verbose:
                print(f'Epoch #{e+1}: MSE Loss = {epoch_loss:.6f}')
            # Save
            torch.save(self.model, self.model_path)


    def predict(self, test_input) -> torch.Tensor:
        '''
        :test_input: tensor of size (N1, C, H, W) with values in range 0-255 that has to
        be denoised by the trained or the loaded network.

        :returns a tensor of size (N1, C, H, W) with values in range 0-255
        '''
        return self.model.forward(test_input).mul(255)