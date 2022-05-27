import torch
import pickle
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

from .others.container import Sequential
from .others.optimizer import SGD
from .others.loss import MSE
from .others.convolution import Conv2d, Upsampling
from .others.activations import ReLU, Sigmoid

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
            Conv2d(3, 64, kernel_size=(3, 3), stride=2, padding=1), 
            ReLU(),
            Conv2d(64, 256, kernel_size=(3, 3), stride=2, padding=1),
            ReLU(),
            Upsampling(256, 64, kernel_size=(4, 4), stride=2, padding=1), 
            ReLU(),
            Upsampling(64, 3, kernel_size=(4, 4), stride=2, padding=1),
            Sigmoid()
        )

        # Default mini batch size
        self.batch_size = 16

        # Default learning rate
        self.learning_rate = 1e-4

        # Instantiate optimizer
        self.optimizer = SGD(self.model.param(), learning_rate=self.learning_rate)

        # Choose loss function
        self.criterion = MSE()

        # Default path to save model
        self.model_path = Path(__file__).parent / "bestmodel.pth"

    def load_pretrained_model(self) -> None:
        '''This loads the parameters saved in bestmodel.pth into the model'''
        # Load parameters
        with open(self.model_path, 'rb') as f:
            params = pickle.load(f)
        # Update parameters
        for i in range(len(self.optimizer.params)):
            self.optimizer.params[i][0].zero_().add_(params[i])

    def train(self, train_input: torch.Tensor, train_target: torch.Tensor, 
        num_epochs=50, verbose=False) -> None:
        ''' Train model

        :train_input: tensor of size (N, C, H, W) containing a noisy version of the images

        :train_target: tensor of size (N, C, H, W) containing another noisy version of the same
        images, which only differs from the input by their noise

        :num_epochs: (int) number of training epochs, default = 50

        :verbose: (bool) Choice of verbose execution, default = False
        '''
        # Scale input and target
        scaled_input = train_input / 255.
        scaled_target = train_target / 255.

        # Train for num_epochs
        for e in range(num_epochs):
            epoch_loss = 0.0
            for inputs, targets in zip(scaled_input.split(self.batch_size), 
                                       scaled_target.split(self.batch_size)):
                output = self.model.forward(inputs)
                epoch_loss += self.criterion.forward(output, targets)
                self.optimizer.zero_grad()
                d_loss = self.criterion.backward()
                self.model.backward(d_loss)
                self.optimizer.step()

            if verbose:
                print(f'Epoch #{e+1}: MSE Loss = {epoch_loss/(train_input.shape[0]/self.batch_size):.6f}')

        # Save at the end of the training
        params = [p for p, g in self.model.param()]
        with open(self.model_path, 'wb') as f:
            pickle.dump(params, f)
        #torch.save(self.model, self.model_path)


    def predict(self, test_input) -> torch.Tensor:
        '''
        :test_input: tensor of size (N1, C, H, W) with values in range 0-255 that has to
        be denoised by the trained or the loaded network.

        :returns a tensor of size (N1, C, H, W) with values in range 0-255
        '''
        return self.model.forward(test_input / 255.).mul(255)
