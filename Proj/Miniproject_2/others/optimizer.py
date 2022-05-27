class SGD:
    '''Stochastic gradient descent optimizer'''
    def __init__(self, params, learning_rate=0.05, weight_decay=0, momentum=0, dampening=0):
        '''SGD optimizer constructor
        
        :params: (iterable) - iterable of parameters to optimize

        :learning_rate: (float) - learning rate, default = 0.05
        '''
        self.params = params
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.velocity = list()

    def zero_grad(self):
        '''Sets the gradients of all optimized torch.Tensor to 0'''
        for _, g in self.params:
            g.zero_()

    def step(self):
        '''Performs single optimization step'''

        # Add weight decay
        ### bizarre
        if (self.weight_decay != 0):
            for p, g in self.params:
                g = g + self.weight_decay * p
        
        # Add momentum
        if (self.momentum != 0):
            if (~self.velocity):
                for _, g in self.params:
                    self.velocity.append(g) 
            else:
                for i, (p, g) in enumerate(self.params):
                    self.velocity[i] = self.momentum * self.velocity[i] + (1 - self.dampening) * g
            
            for i, (p, g) in enumerate(self.params):
                g = self.velocity[i]

        # Update parameters
        for p, g in self.params:
            p -= self.lr * g