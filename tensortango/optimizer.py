import tensortango.mlp as mlp

#Optimizer controls the learning rate

class Optimizer():
    """This class creates an optimizer for the network"""
    def __init__(self, neural_network: mlp.MLP, learning_rate: float = 0.01):
        
        self.net = neural_network
        self.lr = learning_rate

    def step(self):
        raise NotImplementedError
    

class SGD(Optimizer):
    """This class makes a stochasitc gradient descent optimizer"""
    def step(self):
        for param, grad in self.net.params_and_grads():
            param -= grad*self.lr