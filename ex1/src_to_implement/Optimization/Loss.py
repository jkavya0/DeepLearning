import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        
        self.input_tensor = None
        self.label_tensor = None
        self.forward_output = None
        self.backward_output = None

    def forward(self, prediction_tensor, label_tensor):
        """
        Computes the cross-entropy loss accumulated over the batch.
        """
        self.label_tensor = label_tensor

        # Add epsilon to avoid log(0)
        eps = np.finfo(float).eps
        self.input_tensor = prediction_tensor + eps
        
        # Element-wise product to select predicted class probabilities, then log and sum
        log_probs = np.log(np.sum(self.input_tensor * label_tensor, axis=1))
        self.forward_output = -np.sum(log_probs)

        return self.forward_output

    def backward(self, label_tensor):
        """
        Computes the gradient of cross-entropy loss w.r.t the input.
        """
        eps = np.finfo(float).eps
        self.backward_output = - (label_tensor / (self.input_tensor + eps))
        return self.backward_output
