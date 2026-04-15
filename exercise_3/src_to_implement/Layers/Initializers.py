import numpy as np


class Constant:

    def __init__(self, const_val=0.1):
        
        self.const_val = const_val
        self.weights = None

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        
        self.weights = np.full(weights_shape, self.const_val)

        return self.weights


class UniformRandom:

    def __init__(self):
        
        self.weights = None

    def initialize(self, weights_shape, fan_in=None, fan_out=None):

        self.weights = np.random.rand(*weights_shape)
        
        return self.weights


class Xavier:

    def __init__(self):
        
        self.weights = None

    def initialize(self, weights_shape, fan_in, fan_out):
        
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.weights = np.random.normal(0.0, std, size=weights_shape)

        return self.weights


class He:

    def __init__(self):
        
        self.weights = None

    def initialize(self, weights_shape, fan_in, fan_out=None):
        
        std = np.sqrt(2.0 / fan_in)
        self.weights = np.random.normal(0.0, std, size=weights_shape)

        return self.weights