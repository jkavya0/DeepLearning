import numpy as np
from optimizers import Sgd

sgd = Sgd(learning_rate=0.1)

weights = np.array([1.0, 2.0, 3.0])
gradients = np.array([0.1, 0.2, 0.3])

updated_weights = sgd.calculate_update(weights, gradients)
print("Updated weights:", updated_weights)
