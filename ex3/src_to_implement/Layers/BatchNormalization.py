import numpy as np
from Layers.Base import BaseLayer
import copy


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
        self.mean_test = 0
        self.variance_test = 0

        self.channels = channels
        self.weights, self.bias = self.initialize()
        self.forward_output = None
        self.backward_output = None
        self.input_tensor = None
        self.input_tensor_normalized = None
        self.epsilon = 1e-15
        self.momentum = 0.8

        self.entry =0

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
        # is_not_convolutional = len(input_tensor.shape) == 2
        if self.is_convolutional:
            input_tensor = self.reformat(input_tensor)

        if self.testing_phase:

            # use mean test calculated in training time
            self.input_tensor_normalized = (input_tensor - self.mean_test) / np.sqrt(self.variance_test + self.epsilon)
            # Calculate forward output
            if self.is_convolutional:
                self.forward_output = self.reformat(self.input_tensor_normalized * self.weights + self.bias)
            else:
                self.forward_output = self.input_tensor_normalized * self.weights + self.bias

        else:

            # Calculate mean and variance
            self.mean_train = np.mean(input_tensor, axis=0)
            self.variance_train = np.var(input_tensor, axis=0)

            # Calculate mean and variance test
            if self.entry == 0:
                self.mean_test = self.mean_train
                self.variance_test = self.variance_train
                self.entry = 1

            self.mean_test = self.momentum * self.mean_test + (1 - self.momentum) * self.mean_train
            self.variance_test = self.momentum * self.variance_test + (1 - self.momentum) * self.variance_train

            # Normalize input
            self.input_tensor_normalized = (input_tensor - self.mean_train) / np.sqrt(
                self.variance_train + self.epsilon)

            # Calculate forward output
            if self.is_convolutional:
                self.forward_output = self.reformat(self.input_tensor_normalized * self.weights + self.bias)
            else:
                self.forward_output = self.input_tensor_normalized * self.weights + self.bias

        return self.forward_output

    def backward(self, error_tensor):
        # Calculate gradient weights and bias
        input_tensor = self.input_tensor

        if self.is_convolutional:
            error_tensor = self.reformat(error_tensor)
            input_tensor = self.reformat(input_tensor)

        self.gradient_weights = np.sum(error_tensor * self.input_tensor_normalized, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        gradient_input_normalized = error_tensor * self.weights
        # backward output
        gradient_variance = np.sum(gradient_input_normalized * (input_tensor - self.mean_train) * (-1 / 2) *
                                   (self.variance_train + self.epsilon) ** (-3 / 2), axis=0)

        gradient_mean = np.sum(gradient_input_normalized * (-1 / np.sqrt(self.variance_train + self.epsilon)), axis=0)

        self.backward_output = gradient_input_normalized * (
                1 / np.sqrt(self.variance_train + self.epsilon)) + gradient_variance * (
                                       2 / error_tensor.shape[0]) * (
                                       input_tensor - self.mean_train) + gradient_mean * (
                                       1 / error_tensor.shape[0])

        if self.is_convolutional:
            self.backward_output = self.reformat(self.backward_output)

        # Update weights and bias
        if self.optimizer is not None:
            self.weights = self.weight_optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return self.backward_output

    def initialize(self, weights_initializer=None, bias_initializer=None):

        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
        return self.weights, self.bias

    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            return np.concatenate(tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3]),
                                  axis=1).T
        elif len(tensor.shape) == 2:
            return np.transpose(
                tensor.reshape(self.input_tensor.shape[0], self.input_tensor.shape[2] * self.input_tensor.shape[3],
                               self.input_tensor.shape[1]), (0, 2, 1)).reshape(self.input_tensor.shape[0],
                                                                               self.input_tensor.shape[1],
                                                                               self.input_tensor.shape[2],
                                                                               self.input_tensor.shape[3])
        else:
            return tensor