# Layers/Pooling.py

import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):
    """
    2D max‑pooling layer with valid (no) padding.
    Constructor:
        Pooling(stride_shape, pool_shape)
    - stride_shape: int or tuple (sH, sW)
    - pool_shape: tuple (pH, pW)
    """
    def __init__(self, stride_shape, pool_shape):
        super().__init__()             # trainable=False by default
        # normalize stride
        if isinstance(stride_shape, int):
            self.stride = (stride_shape, stride_shape)
        else:
            self.stride = tuple(stride_shape)
        # pooling window
        self.pool_shape = tuple(pool_shape)
        # placeholders
        self.input = None
        self.max_indices = None

    def forward(self, x):
        """
        x: ndarray of shape (N, C, H, W)
        returns: ndarray of shape (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        pH, pW = self.pool_shape
        sH, sW = self.stride

        # “valid” output dims
        H_out = (H - pH) // sH + 1
        W_out = (W - pW) // sW + 1

        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        self.max_indices = np.zeros((N, C, H_out, W_out, 2), dtype=int)
        self.input = x

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h0 = i * sH
                        w0 = j * sW
                        window = x[n, c, h0:h0+pH, w0:w0+pW]
                        idx = np.unravel_index(np.argmax(window), window.shape)
                        self.max_indices[n, c, i, j] = idx
                        out[n, c, i, j] = window[idx]
        return out

    def backward(self, grad_out):
        """
        grad_out: ndarray of shape (N, C, H_out, W_out)
        returns: gradient wrt input, shape (N, C, H, W)
        """
        x = self.input
        N, C, H, W = x.shape
        pH, pW = self.pool_shape
        sH, sW = self.stride
        _, _, H_out, W_out = grad_out.shape

        grad_x = np.zeros_like(x)
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h0 = i * sH
                        w0 = j * sW
                        idx_h, idx_w = self.max_indices[n, c, i, j]
                        grad_x[n, c, h0+idx_h, w0+idx_w] += grad_out[n, c, i, j]
        return grad_x
