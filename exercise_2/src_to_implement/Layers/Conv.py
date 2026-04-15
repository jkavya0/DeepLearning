import numpy as np
import copy
from Layers.Base import BaseLayer

class Conv(BaseLayer):
    """
    A 1D or 2D convolutional layer with 'same' padding.
    Constructor:
        Conv(stride_shape, convolution_shape, num_kernels)
    - stride_shape: int or tuple (sY[, sX])
    - convolution_shape: tuple (C, k) for 1D or (C, kH, kW) for 2D
    - num_kernels: number of output channels (filters)
    """
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()  
        # 1) Store stride and kernel shape
        if isinstance(stride_shape, int):
            if len(convolution_shape) == 2:
                self.stride_shape = (stride_shape,)
            else:
                self.stride_shape = (stride_shape, stride_shape)
        else:
            self.stride_shape = tuple(stride_shape)

        self.convolution_shape = tuple(convolution_shape)  # (C, k) or (C, kH, kW)
        self.num_kernels       = num_kernels

        # 2) Mark as trainable
        self.trainable = True

        # 3) Zero‑mean Gaussian init (σ=0.1)
        C = self.convolution_shape[0]
        if len(self.convolution_shape) == 3:
            # 2D convolution
            _, kH, kW = self.convolution_shape
            self.weights = np.random.randn(
                self.num_kernels, C, kH, kW
            ) * 0.1
        else:
            # 1D convolution
            _, k = self.convolution_shape
            self.weights = np.random.randn(
                self.num_kernels, C, k
            ) * 0.1

        # bias: one value per output channel
        self.bias = np.random.randn(self.num_kernels) * 0.1

        # 4) Placeholders for gradients, input, and optimizers
        self.gradient_weights = None
        self.gradient_bias    = None
        self.input            = None
        self.w_optimizer      = None
        self.b_optimizer      = None

        # 5) Precompute “same” padding amounts
        if len(self.convolution_shape) == 3:
            _, kH, kW = self.convolution_shape
            pad_y, pad_x = kH - 1, kW - 1
            self.pads = [
                (pad_y // 2, pad_y - pad_y // 2),
                (pad_x // 2, pad_x - pad_x // 2)
            ]
        else:
            _, k = self.convolution_shape
            pad = k - 1
            self.pads = [(pad // 2, pad - pad // 2)]

    def initialize(self, w_init, b_init):
        """
        Re‑initialize weights & bias using the provided Initializer objects.
        """
        C = self.convolution_shape[0]
        spatial = int(np.prod(self.convolution_shape[1:]))
        fan_in  = C * spatial
        fan_out = self.num_kernels * spatial

        # weight shape: (F, C, ...kernel dims)
        shape_w = (self.num_kernels, C) + tuple(self.convolution_shape[1:])
        self.weights = w_init.initialize(shape_w, fan_in, fan_out)

        # bias shape: (F,)
        self.bias = b_init.initialize((self.num_kernels,), fan_in, fan_out)

    def forward(self, x):
        self.input = x
        pad_width = [(0,0), (0,0)] + self.pads
        x_p = np.pad(x, pad_width, mode='constant', constant_values=0)

        spatial_dims = x.shape[2:]
        out_dims = []
        for i, dim in enumerate(spatial_dims):
            total = dim + sum(self.pads[i])
            k     = self.convolution_shape[1 + i]
            s     = self.stride_shape[i]
            out_dims.append((total - k) // s + 1)

        N = x.shape[0]
        F = self.num_kernels
        out = np.zeros((N, F) + tuple(out_dims), dtype=x.dtype)

        for n in range(N):
            for f in range(F):
                for idx in np.ndindex(*out_dims):
                    starts = [idx[i] * self.stride_shape[i]
                              for i in range(len(idx))]
                    slices = [n, slice(None)] + [
                        slice(st, st + self.convolution_shape[1+i])
                        for i, st in enumerate(starts)
                    ]
                    region = x_p[tuple(slices)]
                    out[(n, f) + idx] = np.sum(region * self.weights[f]) + self.bias[f]
        return out

    def backward(self, grad_out):
        x = self.input
        pad_width = [(0,0), (0,0)] + self.pads
        x_p = np.pad(x, pad_width, mode='constant', constant_values=0)

        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias    = np.zeros_like(self.bias)
        grad_x_p = np.zeros_like(x_p)

        N = x.shape[0]
        F = self.num_kernels
        out_dims = grad_out.shape[2:]

        for n in range(N):
            for f in range(F):
                for idx in np.ndindex(*out_dims):
                    err = grad_out[(n, f) + idx]
                    starts = [idx[i] * self.stride_shape[i]
                              for i in range(len(idx))]
                    slices = [n, slice(None)] + [
                        slice(st, st + self.convolution_shape[1+i])
                        for i, st in enumerate(starts)
                    ]
                    region = x_p[tuple(slices)]
                    self.gradient_weights[f] += region * err
                    self.gradient_bias[f]    += err
                    grad_x_p[tuple(slices)]  += self.weights[f] * err

        # un‑pad
        unpad = [slice(None), slice(None)]
        for i, dim in enumerate(x.shape[2:]):
            left = self.pads[i][0]
            unpad.append(slice(left, left + dim))
        grad_x = grad_x_p[tuple(unpad)]

        # **correct** parameter update
        if self.w_optimizer is not None:
            self.weights = self.w_optimizer.calculate_update(
                self.weights, self.gradient_weights
            )
        if self.b_optimizer is not None:
            self.bias    = self.b_optimizer.calculate_update(
                self.bias,    self.gradient_bias
            )

        return grad_x

    @property
    def optimizer(self):
        return self.w_optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self.w_optimizer = opt
        self.b_optimizer = copy.deepcopy(opt) if opt is not None else None
