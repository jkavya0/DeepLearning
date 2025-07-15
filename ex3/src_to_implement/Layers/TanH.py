import numpy as np
from Layers.Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self._TanH = None  

    def forward(self, inputs):
        self._TanH = np.tanh(inputs)
        return self._TanH

    def backward(self, delta):
        # Derivative of tanh(x) is 1 - tanh(x)^2
        return delta * (1 - self._TanH ** 2)
