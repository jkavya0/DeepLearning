

import numpy as np

class BaseLayer:
    def __init__(self):
        self.trainable = False
        #self.weights = None 
        self.testing_phase = False  # False means training mode
        self.norm_sum = 0.0  #for regularization tracking 
