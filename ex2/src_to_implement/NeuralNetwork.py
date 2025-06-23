import copy

class NeuralNetwork:
    """
    A class representing a basic feedforward neural network. It supports training 
    via forward and backward propagation and handles both trainable and non-trainable layers.
    """

    def __init__(self, optimizer_value, weights_initializer, bias_initializer):
        """
        Initialize the neural network.
        - optimizer: The optimizer to use for trainable layers.
        Member Variables:
        - loss: A list to store loss values from each training iteration.
        - layers: A list of layers forming the network architecture.
        - data_layer: Provides input and label tensors.
        - loss_layer: Computes the loss and prediction.
        - label_tensor: Temporarily stores ground-truth labels for training.
        """
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer_value
        self.loss = list()
        self.data_layer = None
        self.layers = list()
        self.loss_layer = None

    def forward(self):
        """
        Perform a forward pass through the network:
        - Retrieves input and label from the data layer.
        - Propagates the input through all layers.
        - Calculates and stores the loss via the loss layer.

        Returns:
        - The computed loss value for the current iteration.
        """
        input_tensor, self.label_tensor = self.data_layer.next()

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        loss_value = self.loss_layer.forward(input_tensor, self.label_tensor)

        self.loss.append(loss_value)

        return loss_value

    def backward(self):
        """
        Perform a backward pass through the network:
        - Starts from the loss layer using stored labels.
        - Propagates the gradient backward through all layers.
        """
        gradient = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):

            gradient = layer.backward(gradient)

    def append_layer(self, layer):
        """
        Add a new layer to the network architecture.

        Parameters:
        - layer: The layer object to append.

        If the layer is trainable, a deep copy of the optimizer is set to the layer.
        """
        if layer.trainable:

            layer.optimizer = copy.deepcopy(self.optimizer)

            # Initialize the weights and bias
            layer.initialize(self.weights_initializer, self.bias_initializer)
       
        self.layers.append(layer)

    def train(self, iterations):
        """
        Train the network for a specified number of iterations.

        Parameters:

        - iterations: Number of training cycles (forward + backward).
        """
        for i in range(iterations):

            self.forward()
            self.backward()

    def test(self, input_tensor):
        """
        Evaluate the network on a given input tensor (inference mode).

        Parameters:
        - input_tensor: The input to propagate through the network.

        Returns:
        - The network's prediction (typically from the SoftMax layer).
        """
        for layer in self.layers:

            input_tensor = layer.forward(input_tensor)

        return input_tensor
