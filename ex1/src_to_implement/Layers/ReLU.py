import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):

    def __init__(self):
        
        super().__init__()
        
        self.forward_output = None
        self.backward_output = None
        

    def forward(self, input_tensor):
        
        self.forward_output = np.maximum(0, input_tensor)
        
        return self.forward_output
    

    def backward(self, error_tensor):

        self.backward_output = error_tensor * (self.forward_output > 0)
        
        return self.backward_output
