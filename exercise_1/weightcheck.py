import numpy as np

input_size = 1   # Number of input features (e.g., from previous layer)
output_size = 2  # Number of neurons in this layer

rows = input_size + 1  # +1 for bias
cols = output_size

weights = np.random.rand(rows, cols)

print("Shape of weight matrix:", weights.shape)
print("Weights:\n", weights)
