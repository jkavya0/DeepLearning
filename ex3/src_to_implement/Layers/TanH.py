
import numpy as np
from Layers.Base import BaseLayer


class TanH(BaseLayer):

    def __init__(self):
        super().__init__()
        self.backward_output = None
        self.forward_output = None

    def forward(self, input_tensor):
        self.forward_output = np.tanh(input_tensor)
        return self.forward_output

    def backward(self, error_tensor):
        self.backward_output = error_tensor * (1 - np.square(self.forward_output))
        return self.backward_output
