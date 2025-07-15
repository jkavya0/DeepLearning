import numpy as np
from Layers import TanH, Sigmoid, FullyConnected
from Layers.Base import BaseLayer  # BaseLayer is the base class for all layers
import warnings
import copy

# Simple default weight initializer (Xavier uniform)
class DefaultWeightsInitializer:
    def initialize(self, shape, fan_in, fan_out):
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)

# Simple default bias initializer (zeros)
class DefaultBiasInitializer:
    def initialize(self, shape, fan_in, fan_out):
        return np.zeros(shape)


class RNN(BaseLayer):

    def __init__(self, input_size, hidden_size, output_size):
        
        super().__init__()
        self.norm_sum = 0
        self.hidden_state_int = None
        self.gradient_weights_y = None
        self.optimizer = None
        self._optimizer2 = None
        self.h_value = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize fully connected layers
        self.h_layer = FullyConnected.FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.y_layer = FullyConnected.FullyConnected(self.hidden_size, self.output_size)

        self._optimizer = None
        self.delta_h = None
        self.delta_y = None
        self.backward_output = None
        self.error_tensor = None

        self.time_step = None
        self._weights = None
        self.forward_output = np.array([])
        self._gradient_weights = None
        self._memorize = False

        self.tanH = TanH.TanH()
        self.sigmoid = Sigmoid.Sigmoid()

        self.trainable = True

        self.batch_size = None

        self.input_tensor = None
        self.hidden_state = None

        # Removed calling initialize() here to allow passing initializers explicitly before initialization.

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize_value):
        self._memorize = memorize_value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_value):
        self._optimizer = optimizer_value
        self._optimizer1 = copy.deepcopy(optimizer_value)
        self._optimizer2 = copy.deepcopy(optimizer_value)

    @property
    def weights(self):
        return self.h_layer.weights

    @weights.setter
    def weights(self, weights):
        self.h_layer.weights = weights

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        self.time_step = input_tensor.shape[0]

        self.forward_output = np.zeros((self.time_step, self.output_size))

        # Memorize the hidden state
        if not self.memorize or self.hidden_state is None:
            self.hidden_state = np.zeros((self.time_step, self.hidden_size))
            self.hidden_state_int = np.zeros((1, self.hidden_size))

        for i in range(self.time_step):

            x_new = np.concatenate((
                input_tensor[i].reshape(input_tensor[i].size, 1),
                self.hidden_state_int.reshape(self.hidden_state[i, :].size, 1)
            )).T

            # Calculate the output of h layer
            self.h_value = self.h_layer.forward(x_new)
            self.hidden_state_int = self.tanH.forward(self.h_value.T).reshape(self.hidden_state[i, :].size)
            self.hidden_state[i, :] = self.hidden_state_int

            # Calculate the output of y layer
            y_value = self.y_layer.forward(self.hidden_state[i, :].reshape(1, self.hidden_state[i, :].size))
            self.forward_output[i, :] = self.sigmoid.forward(y_value)

            if self.optimizer is not None and self.optimizer.regularizer is not None:
                self.norm_sum = self.optimizer.regularizer.norm(self.weights)

        return self.forward_output

    def backward(self, error_tensor):

        self.error_tensor = error_tensor

        self.gradient_weights = 0
        self.gradient_weights_y = 0
        self.backward_output = np.zeros((self.time_step, self.input_size))
        h_next = 0

        for i in range(self.time_step - 1, -1, -1):

            # Calculate the Loss of y with respect to the sigmoid function
            self.sigmoid.forward_output = self.forward_output[i, :]
            Loss_y = self.sigmoid.backward(self.error_tensor[i, :])

            # Prepare input tensor for y_layer backward
            self.y_layer.input_tensor = np.concatenate((
                self.hidden_state[i, :],
                np.ones(1)
            )).reshape(1, self.hidden_state[i, :].size + 1)

            # Backprop through y_layer
            self.delta_y = self.y_layer.backward(Loss_y.reshape(1, Loss_y.size))

            # Backprop through tanh
            self.tanH.forward_output = self.hidden_state[i, :]
            Loss_h = self.tanH.backward(self.delta_y + h_next)

            # Prepare input tensor for h_layer backward
            if i == 0:
                tmp_value = np.concatenate((
                    self.input_tensor[i, :],
                    np.zeros(self.hidden_size)
                )).reshape(1, self.input_tensor[i, :].size + self.hidden_size)
            else:
                tmp_value = np.concatenate((
                    self.input_tensor[i, :],
                    self.hidden_state[i-1, :]
                )).reshape(1, self.input_tensor[i, :].size + self.hidden_state[i, :].size)

            self.h_layer.input_tensor = np.concatenate((tmp_value, np.ones((tmp_value.shape[0], 1))), axis=1)

            # Backprop through h_layer
            self.delta_h = self.h_layer.backward(Loss_h)

            # Accumulate gradients for all timesteps
            self.gradient_weights += self.h_layer.gradient_weights
            self.gradient_weights_y += self.y_layer.gradient_weights

            # Calculate error propagated to input
            self.backward_output[i, :] = self.delta_h[:, 0:self.input_size]

            h_next = self.delta_h[:, self.input_size:self.input_size + self.hidden_size]

        if self.optimizer is not None:
            # Apply optimizer update to weights
            self.h_layer.weights = self.optimizer.calculate_update(self.h_layer.weights, self.gradient_weights)
            self.y_layer.weights = self.optimizer.calculate_update(self.y_layer.weights, self.gradient_weights_y)

        return self.backward_output

    def initialize(self, weights_initializer=None, bias_initializer=None):

        # Provide default initializers if none supplied
        if weights_initializer is None:
            weights_initializer = DefaultWeightsInitializer()
        if bias_initializer is None:
            bias_initializer = DefaultBiasInitializer()

        # Initialize the weights of the h layer
        self.h_layer.initialize(weights_initializer, bias_initializer)
        # Initialize the weights of the y layer
        self.y_layer.initialize(weights_initializer, bias_initializer)

        self.weights = self.h_layer.weights
        return self.weights
