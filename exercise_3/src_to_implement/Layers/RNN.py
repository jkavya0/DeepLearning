import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid

class DefaultWeightsInitializer:
    def initialize(self, shape, fan_in, fan_out):
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)

class DefaultBiasInitializer:
    def initialize(self, shape, fan_in, fan_out):
        return np.zeros(shape)

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.h_layer = FullyConnected(input_size + hidden_size, hidden_size)
        self.y_layer = FullyConnected(hidden_size, output_size)

        self.tanh    = TanH()
        self.sigmoid = Sigmoid()

        self._memorize = False
        self.hidden_state = None

        self.forward_output = None
        self.input_tensor  = None
        self.time_step     = 0

        self._gradient_weights  = None
        self.gradient_weights_y = None

        self._optimizer = None

        self.trainable = True

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, flag):
        self._memorize = flag

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
        # inject into sub‑layers
        self.h_layer.optimizer = copy.deepcopy(opt)
        self.y_layer.optimizer = copy.deepcopy(opt)

    @property
    def weights(self):
        return self.h_layer.weights

    @weights.setter
    def weights(self, w):
        self.h_layer.weights = w

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gw):
        self._gradient_weights = gw

    def initialize(self, weights_initializer=None, bias_initializer=None):
        if weights_initializer is None:
            weights_initializer = DefaultWeightsInitializer()
        if bias_initializer is None:
            bias_initializer = DefaultBiasInitializer()

        self.h_layer.initialize(weights_initializer, bias_initializer)
        self.y_layer.initialize(weights_initializer, bias_initializer)
        return self.h_layer.weights

    def forward(self, input_tensor):
        self.input_tensor  = input_tensor
        self.time_step     = input_tensor.shape[0]
        self.forward_output = np.zeros((self.time_step, self.output_size))

        # reset or carry state
        if not self.memorize or self.hidden_state is None:
            self.hidden_state = np.zeros((self.time_step, self.hidden_size))
            h_prev = np.zeros((1, self.hidden_size))
        else:
            h_prev = self.hidden_state[-1:].copy()

        for t in range(self.time_step):
            x_t = input_tensor[t].reshape(1, -1)
            xh = np.hstack([x_t, h_prev])     # shape (1, in+hidden)

            # store forward pass to cache input_tensor, etc.
            h_lin = self.h_layer.forward(xh)
            h_act = self.tanh.forward(h_lin)
            self.hidden_state[t:t+1, :] = h_act
            h_prev = h_act

            y_lin = self.y_layer.forward(h_act)
            y_act = self.sigmoid.forward(y_lin)
            self.forward_output[t, :] = y_act

        return self.forward_output

    def backward(self, error_tensor):
        # initialize accumulators
        self._gradient_weights  = np.zeros_like(self.h_layer.weights)
        self.gradient_weights_y = np.zeros_like(self.y_layer.weights)
        dx = np.zeros((self.time_step, self.input_size))
        dh_next = np.zeros((1, self.hidden_size))

        for t in reversed(range(self.time_step)):
            # --- Re-run forward at timestep t to repopulate caches ---
            #  (required for correct gradient-check behavior)
            x_t = self.input_tensor[t].reshape(1, -1)
            if t == 0:
                h_prev = np.zeros((1, self.hidden_size))
            else:
                h_prev = self.hidden_state[t-1:t, :]

            xh = np.hstack([x_t, h_prev])
            # forward through h-layer and tanh
            h_lin = self.h_layer.forward(xh)
            h_act = self.tanh.forward(h_lin)
            # forward through y-layer and sigmoid
            y_lin = self.y_layer.forward(h_act)
            y_act = self.sigmoid.forward(y_lin)
            

            # 1) output backprop
            dy = self.sigmoid.backward(error_tensor[t].reshape(1, -1))
            dh_from_y = self.y_layer.backward(dy)
            self.gradient_weights_y += self.y_layer.gradient_weights

            # 2) add carry from next step
            dh_total = dh_from_y + dh_next

            # 3) hidden backprop
            dh_lin = self.tanh.backward(dh_total)
            dxh    = self.h_layer.backward(dh_lin)
            self._gradient_weights += self.h_layer.gradient_weights

            # 4) split into dx and dh_next
            dx[t, :]  = dxh[:, : self.input_size]
            dh_next   = dxh[:, self.input_size :]

        # optimizer update
        if self._optimizer is not None:
            self.h_layer.weights = self._optimizer.calculate_update(
                self.h_layer.weights, self._gradient_weights
            )
            self.y_layer.weights = self._optimizer.calculate_update(
                self.y_layer.weights, self.gradient_weights_y
            )

        return dx
