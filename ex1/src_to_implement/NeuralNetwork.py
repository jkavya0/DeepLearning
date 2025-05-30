# Neural Network 1
import copy


class NeuralNetwork:
    # __init__ is the constructor class
    def __init__(self, optimizer_value):
        self.value_out = None
        self.value_in = None
        self.optimizer = optimizer_value
        self.loss = list()
        self.data_layer = None
        self.layers = list()
        self.loss_layer = None

    def forward(self):
        self.value_in, self.value_out = self.data_layer.next()
        for layer in self.layers:
            self.value_in = layer.forward(self.value_in)

        self.value_in = self.loss_layer.forward(self.value_in, self.value_out)
        self.loss.append(self.value_in)

        return self.value_in

    def backward(self):
        # value_out = self.value_out
        self.value_out = self.loss_layer.backward(self.value_out)
        for layer in reversed(self.layers):
            self.value_out = layer.backward(self.value_out)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)

        self.layers.append(layer)

    def train(self, iterations):

        for i in range(iterations):

            self.forward()
            self.backward()

    def test(self, input_tensor):

        value_in = input_tensor
        for layer in self.layers:
            value_in = layer.forward(value_in)

        return value_in