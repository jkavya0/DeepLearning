import numpy as np

class Sgd:
    def __init__(self, learning_rate):
        # Check and convert the learning rate to float
        if isinstance(learning_rate, (int, float)):
            self.learning_rate = float(learning_rate)
        else:
            print('\n//////')
            print("Error - Learning rate is not a number")
            print('//////')
            self.learning_rate = None  

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Computes updated weights using the SGD update rule.
        """
        if self.learning_rate is None:
            raise ValueError("Invalid learning rate. Update cannot be performed.")

        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights
