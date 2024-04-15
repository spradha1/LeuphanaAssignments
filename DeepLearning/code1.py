#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
import sklearn.datasets
import torchvision.datasets
from sklearn.svm import LinearSVC

###############################################################################

#
# Task 3
#

# generate toy data
X,y = sklearn.datasets.make_blobs(n_samples=100, n_features=2, centers=2, random_state=3) # random_state=42
# you can change or remove the random_state!

# change y=0 to y=-1
y[y==0] = -1

# subtract the mean of X from every data point => X has a mean of 0
X = X - X.mean(0)

# plot toy data
fig, ax = plt.subplots()
ax.set_aspect(1)
#       first dim., second dim., dots,label
ax.plot(X[y==+1,0], X[y==+1,1], 'ro', label="positive class")
ax.plot(X[y==-1,0], X[y==-1,1], 'bo', label="negative class")
ax.legend()

# compute R
R = # TODO

# compute gamma
svm = LinearSVC(C=1000, loss="hinge", tol=1e-5, random_state=0)
svm.fit(X, y)
margin = 1 / np.sqrt(np.sum(svm.coef_ ** 2))
gamma = margin

# get the separating hyperplane
w = svm.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (svm.intercept_[0]) / w[1]
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin

# plot toy data
fig, ax = plt.subplots()
ax.set_aspect(1)
#       first dim., second dim., dots,label
ax.plot(X[y==+1,0], X[y==+1,1], 'ro', label="positive class")
ax.plot(X[y==-1,0], X[y==-1,1], 'bo', label="negative class")
ax.plot(xx, yy, 'k-')
ax.plot(xx, yy_down, 'k--')
ax.plot(xx, yy_up, 'k--')
ax.plot([0,X[np.argmax(norms),0]], [0,X[np.argmax(norms),1]], 'k-')
circle = plt.Circle((0, 0), R, color='k', fill=False)
ax.add_artist(circle)
ax.legend()

###############################################################################

# add 1 to each data point
X = np.hstack((np.ones(X.shape[0]).reshape(-1,1), X))

# initialize the weight vector w
w = np.array([0.0, -2.0, 1.0])

def f(X, w):
    return np.dot(X, w)

def classifier(X, w):
    return np.sign(f(X, w))

dom = np.linspace(X[:,1].min()-2, X[:,1].max()+2, 10)

def discriminant(x1, w):
# f(x) = w*x+b
#      = w1x1 * w2x2 + b
# f(x) != 0
# solve for x2, i.e. x2 = ...
# x2 = TODO
    x2 = # TODO
    return x2

# show the decision hyperplane for the current w
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.set_xlim(X[:,1].min()-1,X[:,1].max()+1)
ax.set_ylim(X[:,2].min()-1,X[:,2].max()+1)
ax.plot(X[y==+1,1], X[y==+1,2], 'ro', label="positive class")
ax.plot(X[y==-1,1], X[y==-1,2], 'bo', label="negative class")
ax.plot(dom, discriminant(dom, w), 'k-' ) # k is short for black
ax.legend()


# implementation of the perceptron
def perceptron(X, y, w):
    # TODO

    return w, number_of_total_updates

w, total_updates = perceptron(X, y, w)


# show the decision hyperplane for the learned w
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.set_xlim(X[:,1].min()-1,X[:,1].max()+1)
ax.set_ylim(X[:,2].min()-1,X[:,2].max()+1)
ax.plot(X[y==+1,1], X[y==+1,2], 'ro', label="positive class")
ax.plot(X[y==-1,1], X[y==-1,2], 'bo', label="negative class")
ax.plot(dom, discriminant(dom, w), 'k-') # k is short for black
ax.legend()

###############################################################################

# run the perceptron several times with random w initializations
fig, ax = plt.subplots(figsize=(15,10))
ax.set_aspect(1)
ax.set_xlim(X[:,1].min()-1,X[:,1].max()+1)
ax.set_ylim(X[:,2].min()-1,X[:,2].max()+1)
ax.plot(X[y==+1,1], X[y==+1,2], 'ro', label="positive class")
ax.plot(X[y==-1,1], X[y==-1,2], 'bo', label="negative class")
res = []
for j in range(250):
    # random initialization of w
    w = # TODO
    w, total_updates = perceptron(X, y, w)
    res.append(total_updates)
    ax.plot(dom, discriminant(dom, w), 'k-', alpha=0.1) # k is short for black
ax.legend(loc=2)


# set the bound
bound = # TODO
# does the bound hold for all runs on this data set?
print(np.all(res <= bound))

fig, ax = plt.subplots()
ax.hist(res, density=True)


# now do the same experiment but with a wider and tighter corridor

###############################################################################

#
# Task 4
#

mnist = torchvision.datasets.MNIST('./data', download=True)
X = mnist.data
y = mnist.targets

fig, ax = plt.subplots()
ax.imshow(X[0])

# obtain the indices for the data points labelled with 0 or 1
subset_indices = # TODO

# fix the data set
X = X[subset_indices].numpy()
y = y[subset_indices].numpy()
y[y==0] = -1
X = X.reshape(X.shape[0], -1)

# add 1 to each data point
X = np.hstack((np.ones(X.shape[0]).reshape(-1,1), X))

# initialize the weight vector w
w = # TODO

# run the perceptron
w, total_updates = perceptron(X, y, w)

# check the classification error on train data
accuracy = # TODO

# load test data
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
X_test = mnist_test.data
y_test = mnist_test.targets
subset_indices_test = # TODO
X_test = X_test[subset_indices_test].numpy()
y_test = y_test[subset_indices_test].numpy()
y_test[y_test==0] = -1
X_test = X_test.reshape(X_test.shape[0], -1)
X_test = np.hstack((np.ones(X_test.shape[0]).reshape(-1,1), X_test))


# check the classification error on test data
accuracy = # TODO

# show the misclassified instances
misclassified_indices = # TODO
for idx in misclassified_indices:
    fig, ax = plt.subplots()
    ax.imshow # TODO



