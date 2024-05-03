#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
import sklearn.datasets


###############################################################################

#
# Task 6 Part 1 (i)
#

class ReLU:
    def evaluate(self, z):
        return # TODO
    def gradient(self, z):
        return # TODO

class sigmoid:
    def evaluate(self, z):
        return # TODO
    def gradient(self, z):
        return # TODO

#
# Task 6 Part 1 (ii)
#

"""
# The NeuralNetwork class is supposed to work as follows.
# We can define a network, i.e., with 2 input, 3 hidden, and 1 output neurons as
net = NeuralNetwork([2,3,1])
# A forward pass (prediction) can be computed using
net.forward(x)
# and the gradients of all parameters can be obtained via
net.backprop(x, y)
# Finally, the network can be trained by calling
net.train(x, y)
# Other activation functions can be provided when creating the object:
net = NeuralNetwork([2,3,1], activations=[ReLU(), ReLU(), sigmoid()])
"""

# This impl. uses "arbitrary" rules to transform the derivatives to obtain the gradients
class NeuralNetwork:
    # the __init__ method is called when a object is created
    def __init__(self, neurons_per_layer, activations=None):
        # neurons per layer (incl. input and output)
        self.neurons_per_layer = np.array(neurons_per_layer)
        # list holding the weight matrices
        self.weight = []
        # list holding the bias vectors
        self.bias = []
        # list holding the activation functions
        self.activation = []
        # intermediate states s (they get set by each forward pass)
        self._s = []
        # intermediate states x(they get set by each forward pass)
        self._x = []
        for d_in, d_out in zip(neurons_per_layer[:-1], neurons_per_layer[1:]):
            self.weight.append( # TODO
            self.bias.append( # TODO
        for d in neurons_per_layer[:-1]:
            self._s.append(np.zeros((d,1)))
        for d in neurons_per_layer:
            self._x.append(np.zeros((d,1)))
        if activations==None:
            # if no activations list is given, use sigmoid() as default
            for _ in range(len(neurons_per_layer)-1):
                self.activation.append(sigmoid())
        else:
            self.activation = activations

    def forward(self, x):
        # TODO
        # also set the intermediate states _x and _s
        return self._x[-1]

    def loss(self, y_prediction, y):
        # Task 6 Part 2
        return None

    def backprop(self, x, y):
        # Task 6 Part 2
        return None

    def train(self, x, y, eta=.01, iterations=100):
        # Task 6 Part 2
        return None


# you can also overwrite the parameters yourself
# to confirm that your implementation is correct,
# i.e., that it matches what you computed by hand
net = NeuralNetwork([2,3,2])
# set the weights and biases
net.weight[0] = np.array([[.2, -.3], [.1, -1.2], [.4, .3]])
net.bias[0] = np.array([[.3], [-.1], [1.2]])
net.weight[1] = # TODO
net.bias[1] = # TODO
# set the input
x0 = np.array([[1.], [2.]])
# compute the forward pass
net.forward(x0)
# compare the intermediate states
net._s[0]
net._x[1]
net._s[1]
net._x[2]

###############################################################################

#
# Task 7
#

def f(x):
    return np.sin(x)

# number of functions
n = 5
# beginning the the interval
a = 0
# end the the interval
b = 6
# beginning the the interval
x = np.linspace(a, b, n)
# spacing
h = x[1]-x[0]
# biases
b_ = # TODO
# function values
f_ = # TODO
# matrix used in the linear system of equations
A = # TODO
# fit parameters lambda
lambda_ = # TODO


def approx(x):
    return # TODO


# plot the function and the approximation
dom = np.linspace(-1,8,100)
fig, ax = plt.subplots()
ax.plot(dom, f(dom), 'b-')
ax.plot(dom, approx(dom), 'r-')
