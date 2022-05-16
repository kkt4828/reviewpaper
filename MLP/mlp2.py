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

        self.loss = None
        self.learning_rate = None

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

    def _update_w_b(self, index, dw, delta):
        """
        Update weights and biases
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error
        """

        self.w[index] = self.learning_rate * dw
        self.b[index] = self.learning_rate * np.mean(delta, 0)

    def _back_prop(self, z, a, y_true):
        """
        The input dicts keys represent the layers of the net

        a = { 1: x,
              2: f(w1(x) + b1)
              3: f(w2(a2) + b2)
                }
        :param z: (dict) w(x) + b
        :param a: (dict) f(z)
        :param y_true: (array) One hot encoded truth vector
        """

        delta = self.loss.delta(y_true, a[self.n_layers])
        dw = np.dot(a[self.n_layers - 1].T, delta)

        update_params = {
            self.n_layers - 1: (dw, delta)
        }

        for i in reversed(range(2, self.n_layers)):
            delta = np.dot(delta, self.w[i].T * self.activations[i].prime(z[i]))
            dw = np.dot(a[i - 1].T, delta)
            update_params[i - 1] = (dw, delta)

        for k, v in update_params.items():
            self._update_w_b(k, v[0], v[1])

class MSE:
    def __init__(self, activation_fn):
        """
        :param activation_fn: Class object of the activation function
        """
        self.activation_fn = activation_fn

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector
        :param y_pred: (array) Prediction vector
        :return: (Float)
        """
        return np.mean((y_pred - y_true) ** 2)


    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)





