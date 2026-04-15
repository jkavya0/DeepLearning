# Optimizer File

import numpy as np


# Creating the Optimizer Base Class
class Optimizer:

    def __init__(self, learning_rate=0.01):
        if not isinstance(learning_rate, (int, float)):
            raise TypeError("Error - Learning rate must be a number")

        self.learning_rate = float(learning_rate)
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


# Stochastic Gradient Descent
class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            return np.subtract(weight_tensor, self.learning_rate * (gradient_tensor + self.regularizer.calculate_gradient(weight_tensor)))
        else:
            return np.subtract(weight_tensor, self.learning_rate * gradient_tensor)


# SGD with Momentum
class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__(learning_rate)

        if not isinstance(momentum_rate, (int, float)):
            raise TypeError("Error - Momentum rate must be a number")

        self.momentum_rate = float(momentum_rate)
        self.prev_velocity = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        self.prev_velocity = self.momentum_rate * self.prev_velocity - self.learning_rate * gradient_tensor
        return weight_tensor + self.prev_velocity


# Adam Optimizer
class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__(learning_rate)

        if not isinstance(mu, (int, float)):
            raise TypeError("Error - mu β1 must be a number")
        if not isinstance(rho, (int, float)):
            raise TypeError("Error - rho β2 must be a number")

        self.mu = float(mu)
        self.rho = float(rho)

        self.prev_moment = 0
        self.prev_velocity = 0
        self.entry = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.prev_velocity = self.mu * self.prev_velocity + (1 - self.mu) * gradient_tensor
        self.prev_moment = self.rho * self.prev_moment + (1 - self.rho) * (gradient_tensor ** 2)

        fin_v = self.prev_velocity / (1 - self.mu ** self.entry)
        fin_u = self.prev_moment / (1 - self.rho ** self.entry)

        self.entry += 1

        denom = np.sqrt(fin_u) + np.finfo(float).eps

        if self.regularizer is not None:
            reg_grad = self.regularizer.calculate_gradient(weight_tensor)
            return weight_tensor - self.learning_rate * (fin_v / denom + reg_grad)
        else:
            return weight_tensor - self.learning_rate * (fin_v / denom)
