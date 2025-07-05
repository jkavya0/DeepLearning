import numpy as np

from abc import ABC, abstractmethod

# Base class for all optimizers
# This class defines the interface for all optimizers and provides a method to add regularizers.


class Optimizer(ABC):
    def __init__(self, learning_rate=0.01):
        if isinstance(learning_rate, (int, float)):
            self.learning_rate = float(learning_rate)
        else:
            raise TypeError("Learning rate must be a number.")

        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

    @abstractmethod
    def calculate_update(self, weight_tensor, gradient_tensor):
        pass



class Sgd(Optimizer):
    def __init__(self, learning_rate):

        super().__init__(learning_rate)  # Inherits learning_rate and regularizer handling

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Computes updated weights using the SGD update rule.
        Applies regularization if a regularizer is set.
        """
        if self.regularizer is not None:
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)
            
        # Update the weights
        # The update rule is: w = w - learning_rate * gradient

        return weight_tensor - self.learning_rate * gradient_tensor

    

class SgdWithMomentum(Optimizer):

    def __init__(self, learning_rate, momentum_rate):

        super().__init__(learning_rate)  # Inherits learning_rate and regularizer handling

        if not isinstance(momentum_rate, (int, float)):
            raise TypeError("momentumrate must be a number.")

        self.momentum_rate = float(momentum_rate)
        self.previous_velocity = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Computes updated weights using SGD with momentum.
        Adds regularization if set.
        """
        if self.regularizer is not None:
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)

        # Update the previous velocity

        self.previous_velocity = (
            self.momentum_rate * self.previous_velocity
            - self.learning_rate * gradient_tensor
        )

        return weight_tensor + self.previous_velocity


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__(learning_rate)

        if not isinstance(mu, (int, float)):
            raise TypeError("mu must be a number.")
        if not isinstance(rho, (int, float)):
            raise TypeError("rho must be a number.")

        self.mu = float(mu)
        self.rho = float(rho)
        self.previous_moment = 0
        self.previous_velocity = 0
        self.t = 1  # timestep

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Computes updated weights using the Adam optimization rule.
        Applies regularization if set.
        """
        if self.regularizer is not None:
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)

        # Update the previous velocity 
        self.previous_velocity = self.mu * self.previous_velocity + (1 - self.mu) * gradient_tensor

        # Update the previous moment
        self.previous_moment = self.rho * self.previous_moment + (1 - self.rho) * (gradient_tensor ** 2)

        # Bias correction
        m_hat = self.previous_velocity / (1 - self.mu ** self.t)
        v_hat = self.previous_moment / (1 - self.rho ** self.t)

        # Increment the timestep
        self.t += 1
        
        # Update the weights
        return weight_tensor - self.learning_rate * (m_hat / (np.sqrt(v_hat) + np.finfo(float).eps))
