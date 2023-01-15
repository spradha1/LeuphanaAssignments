"""
  Gaussian Discriminant Analysis for 2D data points
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.datasets import make_blobs


# calculating phi(probability) for class=1 [P(y=1)]
def calculate_phi(y): 
  return len(y[y == 1])/len(y)


# calculating the mean vectors for both classes
def calculate_mu(X, y):
  mu0 = (np.sum(X[y == 0], axis=0) / np.sum([y == 0])).reshape(-1, 1)
  mu1 = (np.sum(X[y == 1], axis=0) / np.sum([y == 1])).reshape(-1, 1)
  return mu0, mu1


# calculating the covariance matrix
def calculate_sigma(X, y, mu0, mu1):
  n = X.shape[0]
  sigma = np.zeros((2, 2))
  for i in range(n):
    xi = X[i, :].reshape(2, 1)
    mu_yi = mu1 if y[i] == 1 else mu0
    sigma += (xi - mu_yi).dot((xi - mu_yi).T)
  return 1/n * sigma


def discriminant(x1, theta):
  x2 = -x1*theta[1]/theta[2] - theta[0]/theta[2]
  return x2


def calculate_probabilities(X, mu0, mu1, sigma):
  constant = 1. / (2*np.pi*np.linalg.det(sigma)**0.5)
  p_y0 = [np.exp(-0.5 * np.dot(np.dot((X[i, :].reshape(-1, 1)-mu0).T, np.linalg.inv(sigma)), (X[i, :].reshape(-1, 1)-mu0))) for i in range(X.shape[0])]
  p_y1 = [np.exp(-0.5 * np.dot(np.dot((X[i, :].reshape(-1, 1)-mu1).T, np.linalg.inv(sigma)), (X[i, :].reshape(-1, 1)-mu1))) for i in range(X.shape[0])]
  return constant*np.array(p_y0), constant*np.array(p_y1)


def plot_points_and_line(X, y, theta, domain, mu0, mu1, sigma):
  # plotting the data points
  plt.plot(X[y == 1, 0], X[y == 1, 1], 'bx', alpha=0.3, label="Class: 1", zorder=1)
  plt.plot(X[y == 0, 0], X[y == 0, 1], 'ro', alpha=0.3, label="Class: 0", zorder=1)
  # plotting the probabilities
  w0_range, w1_range = np.meshgrid(domain, domain)
  X_domain = np.array(list(zip(np.ravel(w0_range), np.ravel(w1_range))))
  z_y0, z_y1 = calculate_probabilities(X_domain, mu0, mu1, sigma)
  z_y0, z_y1 = z_y0.reshape(w0_range.shape), z_y1.reshape(w0_range.shape)
  plt.contour(w0_range, w1_range, z_y0, zorder=2)
  plt.contour(w0_range, w1_range, z_y1, zorder=2)
  # plotting the decision boundary
  plt.plot(domain, discriminant(domain, theta), 'k-')
  plt.legend()
  plt.title('GDA result')
  plt.show()


if __name__ == '__main__':
  # generating a 2-class dataset that is not linearly separable
  X, y = make_blobs(250, n_features=2, centers=[(-1, -1), (1, 1)], random_state=33)
  # scaling the input
  X = preprocessing.scale(X)

  # obtaining all the MLE estimates
  phi = calculate_phi(y)
  mu0, mu1 = calculate_mu(X, y)
  sigma = calculate_sigma(X, y, mu0, mu1)

  '''
    drawing the separating line
    Eqn: ln((1-phi)/phi) + (mu0-mu1)^T * sigma^-1 * x + (1/2)*(mu1^T * sigma^-1 * mu1 - mu0^T * sigma^-1 * mu0) = 0
  '''
  S = np.linalg.inv(sigma)
  theta_12 = S.dot(mu1-mu0).flatten()
  w1 = mu0.T.dot(S.dot(mu0))
  w2 = mu1.T.dot(S.dot(mu1))
  theta_0 = 1/2*(w1-w2)[0, 0]-np.log((1-phi)/phi)
  theta = np.array([theta_0, theta_12.item(0), theta_12.item(1)])
  domain = np.linspace(X[:, 0].min(), X[:, 0].max())
  plot_points_and_line(X, y, theta, domain, mu0, mu1, sigma)