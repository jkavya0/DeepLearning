import copy

class NeuralNetwork:
    """
    A class representing a basic feedforward neural network. It supports training 
    via forward and backward propagation and handles both trainable and non-trainable layers.
    """

    def __init__(self, optimizer_value, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer_value
        self.loss = list()
        self.data_layer = None
        self.layers = list()
        self.loss_layer = None
        self._phase = 'train'  # Default phase

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        """
        Set the phase of the network and all layers ('train' or 'test').
        """
        if value not in ['train', 'test']:
            raise ValueError("Phase must be either 'train' or 'test'")
        
        self._phase = value

        # Set the phase for all layers
        for layer in self.layers:
            layer.phase = value  

        # Set the phase for the loss layer if it exists
        if self.loss_layer is not None:
            self.loss_layer.phase = value  

    def forward(self):

        input_tensor, self.label_tensor = self.data_layer.next()

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        loss_value = self.loss_layer.forward(input_tensor, self.label_tensor)

        self.loss.append(loss_value)
        return loss_value

    def backward(self):

        gradient = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def append_layer(self, layer):

        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)

        self.layers.append(layer)

    def train(self, iterations):
        
        self.phase = 'train'
        for i in range(iterations):
            self.forward()
            self.backward()

    def test(self, input_tensor):
        
        self.phase = 'test'
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
