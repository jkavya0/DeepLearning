import copy

class NeuralNetwork:
    """
    A class representing a basic feedforward neural network. It supports
    training via forward/backward propagation, handles dropout/batch‑norm
    phases, and includes L1/L2 regularization in the loss.
    """

    def __init__(self, optimizer_value, weights_initializer, bias_initializer):
        self.weights_initializer = copy.deepcopy(weights_initializer)
        self.bias_initializer    = copy.deepcopy(bias_initializer)
        self.optimizer           = optimizer_value
        self.loss                 = []
        self.data_layer          = None
        self.layers              = []
        self.loss_layer          = None
        self._phase              = 'train'  # 'train' or 'test'

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        if value not in ('train', 'test'):
            raise ValueError("Phase must be 'train' or 'test'")
        self._phase = value
        # propagate to all layers
        for layer in self.layers:
            layer.testing_phase = (value == 'test')
        # also set on loss layer if it uses a phase
        if hasattr(self.loss_layer, 'testing_phase'):
            self.loss_layer.testing_phase = (value == 'test')

    def append_layer(self, layer):
        """
        Add a layer to the network. If it's trainable, give it its own optimizer
        and initialize its parameters.
        """
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def forward(self):
        """
        Perform a training‐mode forward pass on the next batch from data_layer,
        summing data loss and regularization penalty.
        """
        # get next batch
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor

        # ensure training mode
        self.phase = 'train'

        # forward through layers, accumulating reg loss
        reg_loss = 0.0
        activations = input_tensor
        for layer in self.layers:
            activations = layer.forward(activations)
            if self.optimizer.regularizer is not None and layer.trainable:
                # .norm() returns L1 or L2 penalty
                reg_loss += self.optimizer.regularizer.norm(layer.weights)

        # data loss
        data_loss = self.loss_layer.forward(activations, label_tensor)
        total_loss = data_loss + reg_loss

        # record and return
        self.loss.append(total_loss)
        return total_loss

    def backward(self):
        """
        Perform backpropagation from the loss layer through all trainable layers.
        """
        grad = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self, iterations):
        """
        Train for the given number of iterations.
        """
        for _ in range(iterations):
            self.forward()
            self.backward()

    def test(self, input_tensor):
        """
        Run an inference pass in test mode (dropout off, batch‑norm fixed).
        """
        self.phase = 'test'
        activations = input_tensor
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations
