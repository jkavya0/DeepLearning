import numpy as np

class BaseLayer:

    def __init__(self):
        self.trainable = False
        #self.weights = None
        self.testing_phase = False
        self.norm_sum = 0