import numpy as np

from Layers.Base import BaseLayer

from Optimization import optimizers

class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):

        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True

        self.input_tensor = None
        self.error_tensor = None
        
        self._optimizer = None
        self._gradient_weights = None
        
        rows = input_size + 1  # +1 for bias
        cols = output_size
        wieght_shape = rows + cols #for signal matrix multiplication
        self.weights = np.random.uniform(0.0, 1.0, ( wieght_shape)) # Initialize weights uniformly random in the range [0, 1) 
        
        self.forward_output = None
        self.propagated_error = None

        
    def forward(self, input_tensor):
        
 
    # Add bias (column of ones) to input tensor
        bias = np.ones((input_tensor.shape[0], 1))
        self.input_tensor = np.concatenate((input_tensor, bias), axis=1)
    
    # Multiply input with weights
        self.forward_output = np.dot(self.input_tensor, self.weights)
        return self.forward_output

    
    """
    setter & getter property optimizer: 
        sets and returns the protected member _optimizer for this layer
    """
    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_value):
        self._optimizer = optimizer_value

        """
          - For future reasons, property gradient_weights is provided
          - It returns the gradient with respect to the weights, 
            after they have been calculated in the backward-pass
        """
    
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

        
    def backward(self, error_tensor):
    
        self.propagated_error = np.dot(error_tensor, self.weights.transpose())
        self.gradient_weights = np.dot(self.input_tensor.transpose(), error_tensor)
        
        # Don’t perform an update if the optimizer is not set
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return self.propagated_error[:, :-1] # ':' selects all rows, ':-1' all but the last column, Remove the last column (bias part) 