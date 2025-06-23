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
    

class SgdWithMomentum:

    def __init__(self, learning_rate, momentum_rate):

        # Check and convert the rates to float
        if isinstance(learning_rate, (int, float)):
            self.learning_rate = float(learning_rate)
        else:
            print('\n//////')
            print('Error - Learning rate is not a number')
            print('//////')
            self.learning_rate = None

        if isinstance(momentum_rate, (int, float)):
            self.momentum_rate = float(momentum_rate)
        else:
            print('\n//////')
            print('Error - Momentum rate is not a number')
            print('//////')
            self.momentum_rate = None

        self.previous_velocity = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Computes updated weights using the SGD with momentum update rule.
        """
        if self.learning_rate is None or self.momentum_rate is None:
            raise ValueError("Invalid learning or momentum rate. Update cannot be performed.")

        # v = mu * v - lr * grad
        self.previous_velocity = self.momentum_rate * self.previous_velocity - self.learning_rate * gradient_tensor

        # w_new = w + v
        return weight_tensor + self.previous_velocity



class Adam:

    def __init__(self, learning_rate, mu, rho):

        # Check and convert the rates to float

        if isinstance(learning_rate, (int, float)):
            self.learning_rate = float(learning_rate)
        else:
            print('\n//////')
            print('Error - Learning rate is not a number')
            print('//////')
            self.learning_rate = None

        if isinstance(mu, (int, float)):
            self.mu = float(mu)
        else:
            print('\n//////')
            print('Error - mu is not a number')
            print('//////')
            self.mu = None

        if isinstance(rho, (int, float)):
            self.rho = float(rho)
        else:
            print('\n//////')
            print('Error - rho is not a number')
            print('//////')
            self.rho = None

        # Initialize moment and velocity
        self.previous_moment = 0
        self.previous_velocity = 0
        self.t = 1  # timestep

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Computes updated weights using the Adam optimization update.
        """
        if None in (self.learning_rate, self.mu, self.rho):
            raise ValueError("Invalid hyperparameters. Update cannot be performed.")

        # First moment estimate (m) m_t = β₁ * m_{t-1} + (1-β₁) * g_t
        self.previous_velocity = self.mu * self.previous_velocity + (1 - self.mu) * gradient_tensor
        # Second moment estimate (v) v_t = β₂ * v_{t-1} + (1-β₂) * (g_t^2)
        self.previous_moment = self.rho * self.previous_moment + (1 - self.rho) * (gradient_tensor ** 2)

        # Bias-corrected estimates 
        m_hat = self.previous_velocity / (1 - self.mu ** self.t)
        v_hat = self.previous_moment / (1 - self.rho ** self.t)

        # Increment timestep
        self.t += 1

        # Parameter update
        return weight_tensor - self.learning_rate * (m_hat / (np.sqrt(v_hat) + np.finfo(float).eps))
