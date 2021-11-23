import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os
from scipy.special import expit

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

        self.W.append(np.random.randn(input_shape + layers[0].bias, layers[0].size) * np.sqrt(input_shape))
        if layers[0].bias:
            self.W[-1][:, -1] = np.zeros_like(self.W[-1][:, -1])  
        for i in range(len(layers) - 1):
            self.W.append(np.random.randn(layers[i].size + layers[i+1].bias, layers[i+1].size) * np.sqrt(2/layers[i].size) / 100000)
            if layers[i].bias:
                self.W[-1][:, -1] = np.zeros_like(self.W[-1][:, -1])  

    def print_summary(self):
        s = ''
        s += f'Input Shape: {self.input_shape}\n'
        s += f'Layers:\n'
        for l in range(len(self.layers)):
            s += f'\t{l}:\n'
            s += f'\t\tSize: {self.layers[l].size}\n'
            s += f'\t\tBias: {self.layers[l].bias}\n'
            s += f'\t\tFunc: {self.layers[l].activation_func}\n'
            s += f'\t\tShape: {self.W[l].shape}\n'
        
        print(s)
        return s
    
    def save(self, path):
        path = path.rstrip('/\\')
        try:
            os.mkdir(path)

        except OSError as e:
            self.save(input('Invalid path. Enter a different one: '))
            return

        with open(f'{path}/summary.txt', 'w+') as summary:
            summary.write(self.print_summary())

        for i in range(len(self.W)):
            np.save(f'{path}/w_{i}.npy', self.W[i])
    
    def load(self, path):
        for i in range(len(self.W)):
            self.W[i] = np.load(f'{path}/w_{i}.npy')
            
    def predict(self, X):

        # Feed forward
        p = np.atleast_2d(X)

        for w, layer in zip(self.W, self.layers):
            if layer.bias:
                p = np.hstack((p, np.ones([p.shape[0],1])))
            p = p.dot(w)
            p = layer.activation_func(p)

        return p

    def calc_loss(self, X, Y):
        l1 = 0
        l2 = 0

        for w in self.W:
            l1 += np.sum(np.abs(w))
            l2 += np.sum(w ** 2)
        return (
            self.loss(self.predict(X), Y) +
            self.l1_penalty / X.shape[0] * l1 + 
            self.l2_penalty / X.shape[0] * l2
        )

    def get_next_batch(self, X, Y, batch_size):
        for i in np.arange(0, X.shape[0], batch_size):
            yield (X[i:i + batch_size], Y[i:i + batch_size])
    
    def fit(self, X, Y, testX, testY, epochs = 1000, batch_size = 32, learning_rate = 0.01, momentum = 0):
        training_losses = []
        testing_losses = []

        num_examples = X.shape[0]

        for epoch in progressbar(range(epochs)):

            # Randomize the examples
            p = np.arange(X.shape[0])
            np.random.shuffle(p)
            X = X[p]
            Y = Y[p]

            d_W = [0] * len(self.W)

            for (x, y) in self.get_next_batch(X, Y, batch_size):
                
                # Feed forward
                A = [np.atleast_2d(x)]

                for w, layer in zip(self.W, self.layers):
                    if layer.bias:
                        A[-1] = np.hstack((A[-1], np.ones([A[-1].shape[0],1])))
                    net = A[-1].dot(w)
                    out = layer.activation_func(net)

                    A.append(out)
                
                # Backpropagation
                error = A[-1] - y
                D = [self.d_loss(A[-1], y)]

                for l in range(len(self.W) - 1, 0, -1):
                    delta = D[-1]
                    delta = delta.dot(self.W[l].T)
                    delta = delta * self.layers[l].d_activation_func(A[l])
                    if self.layers[l].bias:
                        delta = delta[:, :-1]
                    D.append(delta)
                
                D = D[::-1]

                # Update weights
                for l in range(len(self.W)):
                    d_W[l] = learning_rate * (
                        A[l].T.dot(D[l]) + 
                        2 * self.l2_penalty / num_examples * self.W[l] +
                        self.l1_penalty / num_examples * np.sign(self.W[l]) * self.W[l]
                    ) + momentum * d_W[l]
                    self.W[l] -= d_W[l]
                print(self.calc_loss(X, Y))

            training_losses.append(self.calc_loss(X, Y))
            print(training_losses[-1])
            testing_losses.append(self.calc_loss(testX, testY))

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
    return expit(z)
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

def elu(z):
    return np.where(z > 0, z, (np.exp(z) - 1))
def d_elu(y):
    return np.where(y > 0, 1, y + 1)

if __name__ == '__main__':

    layers = [
        Layer(2, False, elu, d_elu),
        Layer(1, True, sigmoid, sigmoid),
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

    # Save model
    import time
    path = f'models/xor_{int(time.time())}'
    model.save(path)

    model = FeedforwardNeuralNetwork(2, layers)
    model.load(path)

    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = (np.sum(X, axis=1) % 2)[:, None]

    print(X)
    print(model.predict(X))