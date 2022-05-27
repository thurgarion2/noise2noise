### For mini - project 1
import torch
import torch.nn as nn
import torch.optim as optim
from .others.architectures import UnetWithConcat
from torchvision import transforms
from .others.training_pytorch import ToFloatImage
from pathlib import Path

class Model():
    def __init__( self ) -> None :
        ## instantiate model + optimizer + loss function + any other stuff you need
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = UnetWithConcat().to(self.device)
        
        self.batch_size = 128
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        
        mean, std = [0.4810, 0.4451, 0.3920], [0.2757, 0.2655, 0.2756]
        self.inputs_transforms = transforms.Compose([ToFloatImage(), transforms.Normalize(mean =mean, std = std)])
        self.targets_transforms = transforms.Compose([ToFloatImage()])

    def load_pretrained_model(self) -> None :
        ## This loads the parameters saved in bestmodel . pth into the model
        model_path = Path(__file__).parent / "bestmodel.pth"
        params = torch.load(model_path, map_location = lambda storage, loc: storage)
        self.model.load_state_dict(params)
        

    def train(self, train_input, train_target, num_epochs=1) -> None :
        ''' Train model

        :train_input: tensor of size (N, C, H, W) containing a noisy version of the images

        :train_target: tensor of size (N, C, H, W) containing another noisy version of the same
        images, which only differs from the input by their noise

        :num_epochs: (int) number of training epochs, default  = 1
        '''
        with torch.set_grad_enabled(True):
            self.model.to(self.device)
            for e in range(num_epochs):
                for inputs, targets in zip(train_input.split(self.batch_size), 
                                       train_target.split(self.batch_size)):
                  
                    inputs, targets = self.inputs_transforms(inputs.to(self.device)), self.targets_transforms(targets.to(self.device))
                
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                  



    def predict(self, test_input) -> torch.Tensor:
        # : test ̇input : tensor of size ( N1 , C , H , W ) with values in range 0 - 255 that has to be denoised by the trained or the loaded network .
        # : returns a tensor of the size ( N1 , C , H , W ) with values in range 0 - 255.
        with torch.inference_mode():
            self.model.to("cpu")
            test_input = self.inputs_transforms(test_input)
            
            ##model return values in range 0-1, simpler to compute psnr and we can use sigmoid
            return self.model(test_input)*255
    
