import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from collections import namedtuple
from progressbar import progressbar

Layer = namedtuple('Layer', ['size', 'bias', 'activation_func', 'd_activation_func'])
# This class is for a fully-connected feed forward neural network with per layer activation functions
# 
# Inputs:
#
#   layers - a list of Layers

class FeedforwardNeuralNetwork:

    def __init__(self, input_shape, layers, loss=lambda y_hat, y: np.mean((y-y_hat)**2), d_loss=lambda y_hat, y : 2*(y_hat - y), l1_penalty = 0, l2_penalty = 0):
        self.input_shape = input_shape
        self.W = [] # list of weight matrices
        self.layers = layers
        self.loss = loss
        self.d_loss = d_loss
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

        self.W.append(np.random.randn(input_shape + 1, layers[0].size + layers[0].bias))
        for i in range(len(layers) - 1):
            self.W.append(np.random.randn(layers[i].size + layers[i].bias, layers[i+1].size + layers[i+1].bias) / 10000.0)   

    def print_summary(self):
        print('Input Shape: ', self.input_shape)
        for l in range(len(self.layers)):
            print(self.W[l].shape)
            print(self.layers[l].size, self.layers[l].activation_func)
    
    def predict(self, X, add_ones=True):
        p = np.atleast_2d(X)
        if add_ones:
            p = np.hstack((p, np.ones([X.shape[0],1])))
        for w, layer in zip(self.W, self.layers):
            p = layer.activation_func(np.dot(p, w))
        return p

    def calc_loss(self, X, Y, add_ones=False):
        l1 = 0
        l2 = 0

        for w in self.W:
            l1 += np.sum(np.abs(w))
            l2 += np.sum(w ** 2)
        return (
            self.loss(self.predict(X, add_ones=add_ones), Y) +
            self.l1_penalty / X.shape[0] * l1 + 
            self.l2_penalty / X.shape[0] * l2
        )

    def get_next_batch(self, X, Y, batch_size):
        for i in np.arange(0, X.shape[0], batch_size):
            yield (X[i:i + batch_size], Y[i:i + batch_size])
    
    def fit(self, X, Y, testX, testY, epochs = 1000, batch_size = 32, learning_rate = 0.01, momentum = 0):
        X = np.hstack((X, np.ones([X.shape[0], 1])))
        training_losses = []
        testing_losses = []

        num_examples = X.shape[0]

        for epoch in progressbar(range(epochs)):

            # Randomize the examples
            p = np.arange(X.shape[0])
            np.random.shuffle(p)
            X = X[p]
            Y = Y[p]

            for (x, y) in self.get_next_batch(X, Y, batch_size):
                
                # Feed forward
                A = [np.atleast_2d(x)]

                for w, layer in zip(self.W, self.layers):
                    net = A[-1].dot(w)
                    out = layer.activation_func(net)

                    A.append(out)
                
                # Backpropagation
                error = A[-1] - y
                D = [self.d_loss(A[-1], y)]

                for l in range(len(A) - 2, 0, -1):
                    delta = D[-1].dot(self.W[l].T)
                    delta = delta * self.layers[l].d_activation_func(A[l])
                    D.append(delta)
                
                D = D[::-1]

                # Update weights
                for l in range(len(self.W)):
                    self.W[l] -= learning_rate * (
                        A[l].T.dot(D[l]) + 
                        2 * self.l2_penalty / num_examples * self.W[l] +
                        self.l1_penalty / num_examples * np.sign(self.W[l]) * self.W[l]
                    )

            training_losses.append(self.calc_loss(X, Y))
            testing_losses.append(self.calc_loss(testX, testY, add_ones=True))

        fig, ax1 = plt.subplots()

        # plot the losses
        ax1.set_xlabel('Training Epoch')
        ax1.set_ylabel('Loss')
        
        # plot the accuracy
        p1 = ax1.plot(np.arange(0, epochs), training_losses, label = 'Loss (train)', color = 'tab:orange')
        p2 = ax1.plot(np.arange(0, epochs), testing_losses, label = 'Loss (test)', color = 'tab:green')

        # add a legend
        ps = p1 + p2
        labs = [p.get_label() for p in ps]
        ax1.legend(ps, labs, loc=0)
        plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def d_sigmoid(y):
    return y * (1 - y)

def tanh(z):
    return np.tanh(z)
def d_tanh(y):
    return 1 - y**2

def relu(z):
    return np.maximum(0, z)
def d_relu(y):
    return np.sign(y)

def leaky_relu(z):
    return np.maximum(0.1*z, z)
def d_leaky_relu(y):
    return np.maximum(0.1, np.sign(y))

if __name__ == '__main__':

    layers = [
        Layer(2, True, sigmoid, d_sigmoid),
        Layer(1, False, sigmoid, d_sigmoid),
    ]

    X = np.round(np.random.rand(1000, 2))
    Y = (np.sum(X, axis=1) % 2)[:, None]

    model = FeedforwardNeuralNetwork(2, layers)
    model.print_summary()

    model.fit(X, Y, X, Y, learning_rate=0.001)

    fig, ax1 = plt.subplots()

    # plot the losses
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # plot the accuracy
    p1 = ax1.plot(Y, label = 'Data', color = 'tab:orange')
    p2 = ax1.plot(model.predict(X), label = 'Prediction', color = 'tab:green')

    # add a legend
    ps = p1 + p2
    labs = [p.get_label() for p in ps]
    ax1.legend(ps, labs, loc=0)
    plt.show()