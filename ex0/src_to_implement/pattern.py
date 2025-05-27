# Author - Kavya Jayaramaiah
# idmid: iz81eniq

import numpy as np
import matplotlib.pyplot as plt

class Checker:

    def __init__(self, resolution, tile):
       
            self.resolution = resolution
            self.tile = tile
            self.output = np.array([])

    def draw(self):

        w = np.ones([self.tile, self.tile])
        b = np.zeros([self.tile, self.tile])

        board_matrix = np.block([[b,w],[w,b]])
        top = np.hstack((b,w))
        bottom = np.hstack((w,b))
        board_matrix = np.vstack((top, bottom))

        matrix_len = int(self.resolution / (2 * self.tile))
        output = np.tile(board_matrix, [matrix_len, matrix_len])
        self.output = output * 1

        return output

    def show(self):
        plt.imshow(self.draw(), cmap='gray')
        plt.colorbar()
        plt.show()


# Creating a Circle Class

class Circle:
    def __init__(self, resolution, radius, position):
    
            self.resolution = resolution
            self.radius = radius
            self.position = position
            self.output = np.array([])

    def draw(self):

        lin_arr = np.linspace(1, self.resolution, self.resolution * 1)
        x, y = np.meshgrid(lin_arr, lin_arr)
        b = x * 0
        b[np.where((x - self.position[0] - 1) ** 2 + (y - self.position[1] - 1) ** 2 <= self.radius ** 2)] = 1
        w = b * 1
        b = b.astype('bool')
        self.output = w.astype('bool')

        return b

    def show(self):
        plt.imshow(self.draw(), cmap='gray')
        plt.colorbar()
        plt.show()


# Creating a spectrum class

class Spectrum:

    def __init__(self, resolution):
       
            self.resolution = resolution
            self.output = np.array([])

    def draw(self):

        (template_mat, _) = np.meshgrid(np.linspace(0, 1, self.resolution), np.linspace(0, 1, self.resolution))

        red = template_mat
        blue = np.flip(red)
        green = np.rot90(blue)

        fin_rgb = np.dstack((red, green, blue))
        a = fin_rgb * 1
        self.output = a

        return fin_rgb

    def show(self):
        plt.imshow(self.draw())
        plt.show()
