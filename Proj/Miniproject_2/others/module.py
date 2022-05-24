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