import numpy as np
import tensor

class Loss():
    """Abstract base class for loss functions"""

    def loss(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> float:
        raise NotImplementedError
    
    def gradient(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> tensor.Tensor:
        """Gradient is derivate of loss function with respect to each parameter"""
        raise NotImplementedError
    


class MSE(Loss):
    """Mean Square Error Loss Function"""

    def loss(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> float:
        return np.mean((predictions - labels)**2)
    
    def gradient(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> float:
        # First derivative of loss function
        return 2 * (predictions - labels)
    