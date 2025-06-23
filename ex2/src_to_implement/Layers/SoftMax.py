import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):

    def __init__(self):
        
        super().__init__()
        
        self.forward_output = None
        self.backward_output = None
        

    def forward(self, input_tensor):
        
        exp_value = np.exp(input_tensor - input_tensor.max(axis=1, keepdims=True))
        self.forward_output = exp_value / exp_value.sum(axis=1, keepdims=True)

        return self.forward_output
    

    def backward(self, error_tensor):

        dot_product = np.sum(error_tensor * self.forward_output, axis=1, keepdims=True)
        self.backward_output = self.forward_output * (error_tensor - dot_product)
        
        return self.backward_output
