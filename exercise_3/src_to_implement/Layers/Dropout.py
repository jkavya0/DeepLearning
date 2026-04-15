import numpy as np
from Layers.Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None
        self.forward_output = None
        self.backward_output = None

    def forward(self, input_tensor):
        """
        Forward pass for Dropout.
        Uses inverted dropout during training; no dropout during testing.
        Stores the output for debugging or analysis.
        """
        if self.testing_phase:
            # No dropout during testing phase
            self.forward_output = input_tensor
        else:
            # Generate mask and scale for inverted dropout
            self.mask = np.random.binomial(1, self.probability, size=input_tensor.shape)
            self.mask = self.mask / self.probability
            self.forward_output = input_tensor * self.mask

        return self.forward_output

    def backward(self, error_tensor):
        """
        Backward pass for Dropout.
        Multiplies gradient by the same mask used in forward pass.
        """
        self.backward_output = error_tensor * self.mask
        return self.backward_output
