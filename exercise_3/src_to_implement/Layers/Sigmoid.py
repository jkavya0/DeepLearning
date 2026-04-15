import numpy as np
from Layers.Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self._sigmoid_output = None  

    def forward(self, inputs):
        
        inputs = np.clip(inputs, -500, 500)
        self._sigmoid_output = 1 / (1 + np.exp(-inputs))
        return self._sigmoid_output

    def backward(self, delta):
        # Derivative: σ(x)(1 - σ(x))
        return delta * self._sigmoid_output * (1 - self._sigmoid_output)
