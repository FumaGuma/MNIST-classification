import numpy as np
from abc import ABC, abstractmethod

###sotfmax
class Layer(ABC):
    """Layer abstract class"""
    @abstractmethod
    def forward(self, x, training = True):
        pass

    @abstractmethod
    def backward(self, loss_prop):
        pass

    def optimize(self, learning_rate):
        pass

    def describe(self):
        pass

###sotfmax
class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim, initialization = 'kaiming'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w_vel = np.zeros((input_dim, output_dim))
        self.b_vel = np.zeros(output_dim)
        ### Initialize the weight and bias values with Kaiming initialization
        if initialization == 'kaiming':
            var = 1 / input_dim
            self.biases = np.random.uniform(-var, var, output_dim)
        else:
            self.biases = np.random.randn(output_dim)
        if initialization == 'kaiming':
            gain = np.sqrt(2.0)
            std = gain / np.sqrt(input_dim)
            bound = np.sqrt(3.0) * std
            self.weights = np.random.uniform(-bound, bound, (input_dim, output_dim))
        else:
            self.weights = np.random.randn(input_dim, output_dim)

    def forward(self, x, training = True):
        """ Input vector is of the shape (batch_size, input_dim), output is of the shape (batch_size, output_dim)
            Fully connected layer with weights between all of the input and output elements.
        """
        self.previous_activation = x
        x = np.dot(x, self.weights) + self.biases
        return x

    def backward(self, loss_prop):
        self.weight_grad = np.dot(self.previous_activation.T, loss_prop)
        self.bias_grad = np.sum(loss_prop, axis = 0)

        delta = np.dot(loss_prop, self.weights.T)
        return delta

    def optimize(self, learning_rate):
        self.weights -= self.weight_grad * learning_rate
        self.biases -= self.bias_grad * learning_rate

    def describe(self):
        print(f"    Dense layer, input dimensions: {self.input_dim}, output dimensions: {self.output_dim}")
class Sigmoid(Layer):
    def forward(self, x, training = True):
        self.previous_activation = 1/(1+np.exp(-x))
        return self.previous_activation
    def backward(self, loss_prop):
        sig = self.previous_activation
        return loss_prop * sig * (1 - sig)

    def describe(self):
        print(f"    Sigmoid layer")
class ReLu(Layer):
    """ReLu sets the values of inputs elements smaller than 0 to 0, introduces non-linearity"""

    def forward(self, x, training = True):
        mask = x > 0.0
        out = x * mask

        # save cache for back-propagation
        self.cache = mask

        return out

    def backward(self, loss_prop):
        mask = self.cache

        # shut down gradients at negative positions
        dX = loss_prop * mask

        # clear cache
        self.cache = None

        return dX

    def describe(self):
        print(f"    ReLu Layer")
class Dropout(Layer):
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def forward(self, x, training = True):
        """Dropout sets the values of certain percentage of input vectore elements of each pass to 0, used for regularization """
        if training:
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * self.mask / (1 - self.dropout_rate)
        else:
            return x

    def backward(self, loss_prop):
        return loss_prop * self.mask / (1 - self.dropout_rate)

    def describe(self):
        print(f"    Dropout layer, dropout rate: {self.dropout_rate}")

class CrossEntropyLoss():
    def __init__(self, reduction="mean"):
        # add epsilon in log to prevent inf values
        self.epsilon = 1e-8
        self.cache = None

    def forward(self, x, y):
        """Calculate the cross-entropy loss for a batch.

        Args:
            x: inputs batch
            y: labels batch

        Returns:
            loss: scalar cross-entropy loss
        """
        # calculate softmax probabilities
        # https: // stackoverflow.com / questions / 42599498 / numerically - stable - softmax / 42606665  # 42606665
        # softmax function is not changed by adding arbitrary values to all elements of the vector
        # substracting maximum makes all vectors non-positive and protects against overflow
        x = x - np.max(x, axis=1, keepdims=True)
        prob = np.exp(x)
        prob = prob / np.sum(prob, axis=1, keepdims=True)

        loss = -np.sum(y * np.log(prob + self.epsilon))
        m, _ = x.shape
        loss /= m

        # save cache for back-propagation
        self.cache = prob, y

        return loss

    def backward(self):
        prob, y = self.cache

        dX = prob - y
        m = prob.shape[0]
        dX /= m

        self.cache = None
        return dX


class NN():
    def __init__(self):
        self.layers = []
        self.training_time = 0

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x, training='False'):
        for layer in self.layers:
            x = layer.forward(x, training = training)
        return x

    def backprop(self, loss_prop):
        #loss_prop = y - self.network_output
        for layer in reversed(self.layers):
            loss_prop = layer.backward(loss_prop)

    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.optimize(learning_rate)

    def set_training_data(self, training_time, learning_rate, epochs, batch_size):
        self.training_time = training_time
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def get_description(self):
        print(f"Training_time: {round(self.training_time,2)} seconds")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Number of epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print("Model consists of:")
        for layer in self.layers:
            layer.describe()