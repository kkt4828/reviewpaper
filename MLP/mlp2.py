import numpy as np

class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))
    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))

class ReLU:
    @staticmethod
    def activation(z):
        z[z < 0] = 0
        return z



class Network:
    def __init__(self, dimensions, activations):
        self.w = {}
        self.b = {}
        self.dimensions = dimensions
        self.n_layers = len(self.dimensions)

        self.activations = {}

        for i in range(1, self.n_layers):
            self.w[i] = np.random.randn(self.dimensions[i], self.dimensions[i-1]) / np.sqrt(self.dimensions[i-1])
            self.b[i] = np.zeros(self.dimensions[i])
            self.activations[i+1] = activations[i-1]

    def _feed_forward(self, x):

        z = {}
        a = {1 : x}

        for i in range(1, self.n_layers):
            z[i + 1] = np.dot(self.w[i], a[i]) + self.b[i]
            a[i + 1] = self.activations[i+1].activation(z[i + 1])
        return z, a


    def predict(self, x):

        _, a = self._feed_forward(x)
        return a[self.n_layers]










