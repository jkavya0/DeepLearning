import numpy as np
import copy
from Layers.Base import BaseLayer


class BatchNormalization(BaseLayer):

    def __init__(self, channels):
        super().__init__()

        self.is_convolutional = None
        self.trainable = True

        self._gradient_bias = None
        self._gradient_weights = None
        self._bias_optimizer = None
        self._weight_optimizer = None
        self._optimizer = None

        self.variance_train = None
        self.mean_train = None
        self.mean_test = None
        self.variance_test = None

        self.channels = channels
        self.weights = np.ones(channels)     # gamma initialized to ones
        self.bias = np.zeros(channels)       # beta initialized to zeros
        self.forward_output = None
        self.backward_output = None
        self.input_tensor = None
        self.input_tensor_normalized = None

        self.epsilon = 1e-15
        self.momentum = 0.8
        self.initialized = False

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_value):
        # Optimizer is copied for each trainable parameter
        self._optimizer = optimizer_value
        self._weight_optimizer = copy.deepcopy(optimizer_value)
        self._bias_optimizer = copy.deepcopy(optimizer_value)

    @property
    def weight_optimizer(self):
        return self._weight_optimizer

    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.is_convolutional = len(input_tensor.shape) == 4

        # Reformat input if convolutional
        input_reshaped = self.reformat(input_tensor) if self.is_convolutional else input_tensor

        if self.testing_phase:
            # Use mean and variance from training phase (running average)
            mean = self.mean_test
            var = self.variance_test
        else:
            # Calculate mean and variance from current batch
            mean = np.mean(input_reshaped, axis=0)
            var = np.var(input_reshaped, axis=0)

            # Initialize running statistics on first batch
            if not self.initialized:
                self.mean_test = mean
                self.variance_test = var
                self.initialized = True
            else:
                # Update running averages using momentum
                self.mean_test = self.momentum * self.mean_test + (1 - self.momentum) * mean
                self.variance_test = self.momentum * self.variance_test + (1 - self.momentum) * var

            self.mean_train = mean
            self.variance_train = var

        # Normalize input
        self.input_tensor_normalized = (input_reshaped - mean) / np.sqrt(var + self.epsilon)

        # Apply learned scale and shift (gamma and beta)
        output_reshaped = self.input_tensor_normalized * self.weights + self.bias

        # Reformat back to original shape if needed
        self.forward_output = self.reformat(output_reshaped) if self.is_convolutional else output_reshaped

        return self.forward_output

    def backward(self, error_tensor):
        # Reformat inputs if convolutional
        x = self.reformat(self.input_tensor) if self.is_convolutional else self.input_tensor
        error = self.reformat(error_tensor) if self.is_convolutional else error_tensor
        N = error.shape[0]

        # Calculate gradients for gamma (weights) and beta (bias)
        self.gradient_weights = np.sum(error * self.input_tensor_normalized, axis=0)
        self.gradient_bias = np.sum(error, axis=0)

        # Gradient with respect to normalized input
        grad_norm = error * self.weights

        # Gradient with respect to variance
        grad_var = np.sum(grad_norm * (x - self.mean_train) * -0.5 / (np.power(self.variance_train + self.epsilon, 1.5)), axis=0)

        # Gradient with respect to mean
        grad_mean = np.sum(grad_norm * -1 / np.sqrt(self.variance_train + self.epsilon), axis=0)

        # Combine to get gradient with respect to input
        self.backward_output = grad_norm / np.sqrt(self.variance_train + self.epsilon) + \
                               grad_var * 2 * (x - self.mean_train) / N + \
                               grad_mean / N

        # Reformat back to original shape if needed
        if self.is_convolutional:
            self.backward_output = self.reformat(self.backward_output)

        # Update weights and bias using optimizers
        if self.optimizer is not None:
            self.weights = self.weight_optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return self.backward_output

    def initialize(self, weights_initializer=None, bias_initializer=None):
        # gamma = 1, beta = 0
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
        return self.weights, self.bias

    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            # Convert from shape (batch, channel, height, width) to (batch * height * width, channels)
            b, c, h, w = tensor.shape
            return tensor.transpose(0, 2, 3, 1).reshape(b * h * w, c)

        elif len(tensor.shape) == 2:
            # Convert back from (batch * height * width, channels) to (batch, channel, height, width)
            b, c, h, w = self.input_tensor.shape
            return tensor.reshape(b, h, w, c).transpose(0, 3, 1, 2)

        return tensor
