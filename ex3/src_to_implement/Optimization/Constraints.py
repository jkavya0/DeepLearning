
import numpy as np


class L2_Regularizer:

    def __init__(self, alpha: float):
        self.alpha = alpha

    def calculate_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of L2 regularization: alpha * weights.
        """
        return self.alpha * weights

    def norm(self, weights: np.ndarray) -> float:
        """
        Computes the squared L2 norm: alpha * ||weights||^2.
        """
        return self.alpha * np.sum(weights ** 2)


class L1_Regularizer:

    def __init__(self, alpha: float):
        self.alpha = alpha

    def calculate_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Computes the sub-gradient of L1 regularization: alpha * sign(weights).
        """
        return self.alpha * np.sign(weights)

    def norm(self, weights: np.ndarray) -> float:
        """
        Computes the L1 norm: alpha * sum(abs(weights)).
        """
        return self.alpha * np.sum(np.abs(weights))
