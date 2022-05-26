import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

from others.container import Sequential
from others.optimizer import SGD
from others.loss import MSE
from others.convolution import Conv2d, Upsampling
from others.activations import ReLU, Sigmoid

from pathlib import Path

# We turn autograd globally off
torch.set_grad_enabled(False)
# Control randomness for reproducibility
torch.manual_seed(42)

class Model():
    '''Model'''
    def __init__(self) -> None:
        '''Instantiate model + optimizer + loss function + any other stuff needed'''
        # Instantiate model
        self.model = Sequential(
            Conv2d(3, 64, (3,3), stride=2, padding=1), # 10, 15, 15
            ReLU(),
            Conv2d(64, 64, (3,3), stride=2, padding=1), # 20, 7, 7
            ReLU(),
            Upsampling(64, 64, (4,4), stride=2, padding=1), # 10, 15, 15 
            ReLU(),
            Upsampling(64, 3, (4,4), stride=2, padding=1), # 3, 32, 32
            Sigmoid()
        )
        # Instantiate optimizer
        self.optimizer = SGD(self.model.param(), learning_rate=1e-4)

        # Choose loss function
        self.criterion = MSE()

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
                print(f'Epoch #{e+1}: MSE Loss = {epoch_loss/(train_input.shape[0]/self.batch_size):.6f}')

        # Save at the end of the training
        torch.save(self.model, self.model_path)


    def predict(self, test_input) -> torch.Tensor:
        '''
        :test_input: tensor of size (N1, C, H, W) with values in range 0-255 that has to
        be denoised by the trained or the loaded network.

        :returns a tensor of size (N1, C, H, W) with values in range 0-255
        '''
        return self.model.forward(test_input).mul(255)