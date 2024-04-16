import numpy as np
import tensortango.tensor as tensor


class Layer():

    def __init__(self):
        # Weights and biases
        self.w = tensor.Tensor
        self.b = tensor.Tensor

        # Inputs
        self.x = None

        # Gradients
        self.grad_w = 0
        self.grad_b = 0

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """This function computes the feedforward part of the network

        Args:
            x (tensor.Tensor): Inputs into the network
        Returns:
            tensor.Tensor: a numpy ndarray
        """
        raise NotImplementedError
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """Backpropagates through the layer

        Args:
            grad (tensor.Tensor): the gradient of the loss function

        Returns:
            tensor.Tensor: the updated gradient
        """
        raise NotImplementedError
    

class Linear(Layer):

    def __init__(self, input_size: int, output_size: int):
        """Creates a linear layer

        Args:
            input_size (int): number of input values (batch, input)
            output_size (int): number of output values
        """

        #Calling constructor of parent class Layer
        super.__init__()

        #Setting random initial weights and biases
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """Computing y = w @ x + b
        where @ is matrix multiplication (equivalent to scalar *)"""

        self.x = x
        return self.x @ self.w + self.b
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """
        X = w*x + b
        y = f(X)
        dy/dw = f'(X)*x
        dy/dx = f'(X)*w
        dy/db = f'(X)

        The new component being added is the tensor form:
        if y = f(X) and X = x @ w + b and f'(X) is grad
        dy/dx = f'(X) @ w.T
        dy/dw = x.T @ f'(X)
        dy/db = f'(X)
        """
        # Sum along the batch dimension
        self.grad_b = np.sum(grad, axis=0)
        self.grad_w = self.x.T @ grad
        return grad @ self.w.T
    

class Activation(Layer):
        def __init__(self, 
                 input_size: int, 
                 output_size: int,
                 f, 
                 f_prime) -> None:
            
            """Initialize an activation layer as a generic layer that also
            has a function and its derivative, from which the gradient can
            be computed

            Args:
                input_size (int): the number of input values to the layer
                    (batch_size, input_size)
                output_size (int): the number of output values to the next
                    layer (or final value)
                    (batch_size, output_size)
                f (Callable[[tensor.Tensor], tensor.Tensor]): a 
                    differentiable function
                f_prime (Callable[[tensor.Tensor], tensor.Tensor]): the 
                    first derivative of f, as a function
            """
            
            super().__init__()

            self.w = np.random.randn(input_size, output_size)
            self.b = np.random.randn(output_size)

            self.f = f
            self.f_prime = f_prime

        def forward(self, x):
            self.x = x
            #Activation function multiplied by weights and biases
            return self.f(self.x @ self.w + self.b)
        
        def backward(self, grad):
            self.grad_b = np.sum(grad, axis=0)
            self.grad_w = self.x.T @ grad

            grad = grad @ self.w.T

            return self.f_prime(self.x)*grad
    
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    y = tanh(x)
    return 1 - y**2

def relu(x):
    x = np.clip(x, 0, None)
    return x

def relu_prime(x):
    return (x > 0).astype(float)

class Tanh(Activation):
    """Tanh activation function"""

    def __init__(self, input_size, output_size):

        super().__init__(input_size, output_size, tanh, tanh_prime)

class Relu(Activation):
    """Reluactivation function"""

    def __init__(self, input_size, output_size):

        super().__init__(input_size, output_size, relu, relu_prime)
    

