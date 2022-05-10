import numpy as np
class Sigmoid:
    @staticmethod # decorator -> class level에서 직접 불러올 수 있게해주는 기능 => object를 따로 만들어줄 필요가 없음
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
# activations = {1:x, 2: ReLU, 3: Sigmoid}
class Network:
    def __init__(self, dimensions, activations):

        """
        :param dimensions: (tuple / list) Dimensions of the neural net (input, hidden, output)
        :param activations: (tuple / list) Activations functions
        """

        self.n_layers = len(dimensions)
        self.loss = None
        self.learning_rate = None

        self.w = {}
        self.b = {}

        self.activations = {}

        for i in range(len(dimensions) - 1):
            self.w[i+1] = np.random.randn(dimensions[i], dimensions[i + 1]) / np.sqrt(dimensions[i])
            # current layers dim의 sqrt로 나눠주는 것이 Xavier initialization 방법 => 초기값이 너무 크거나 작아지는 것을 막아줌
            self.b[i+1] = np.zeros(dimensions[i + 1])
            self.activations[i+2] = activations[i]
    def _feed_forward(self, x):
        """
        Execute a forward feed through the network

        :param x: (array) Batch of input data vectors
        :return: (tuple) Node outputs and activations per layer
                 The Numbering of the output is equivalent to the layer numbers
        """

        z = {}

        a = {1 : x}
        for i in range(1, self.n_layers):
            # current_layer = i
            # activation_layer = i + 1
            z[i + 1] = np.dot(a[i], self.w[i]) + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])

        return z, a

    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes)
        """
        _, a = self._feed_forward(x)
        return a[self.n_layers] # 제일 마지막 index값이 output 이므로


np.random.seed(2022)
nn = Network((2, 3, 1), (ReLU, Sigmoid))
print('weight : ', nn.w)
print('activations : ', nn.activations)


class MSE:
    def __init__(self, activation_fn):
        """
        :param activation_fn: class object of the activation function
        """

        self.activation_fn = activation_fn


    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector
        :param y_pred: (array) Prediction vector
        :return: (float)
        """

        return np.mean((y_pred - y_true) ** 2) * 0.5
    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

    def delta(self, y_true, y_pred):
        """
        Back Propagation error delta
        :return: (array)
        """

        return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)


