import tensor, layer

class MLP:
    """This class creates a multi-layer perceptron"""

    def __init__(self, layers):
        self.layers = layers


    def forward(self, x):
        """Compute forward pass through entire network"""

        for layer in self.layers:
            x = layer.forward(x)

        return x
    

    def backward(self, grad):
        """Do a backprop pass"""

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad
    

    def params_and_grads(self):
        for layer in self.layers:
            
            #The pair is the weights/biases and their gradients
            for pair in [(layer.w, layer.grad_w), (layer.b, layer.grad_b)]:
                yield pair

    
    def zero_parameters(self):
        """This function resets the gradient values to 0"""
        for layer in self.layers:
            layer.grad_w[:] = 0
            layer.grad_b[:] = 0