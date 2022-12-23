# plot labelled points with separator

import perceptron as pcp
import numpy as np
import matplotlib.pyplot as plt

'''
  get points for the hyperplane
  eqn with slope: -w[0]/w[1] & intercept: -b/w[1]
'''
def get_hyperplane(x1, w, b, x):
  x2 = []
  if w[1] != 0:
    x2 = -x1*w[0]/w[1] - b/w[1]
  elif w[0] != 0:
    x1 = np.full(x1.shape, -b/w[0])
    x2 = np.linspace( min(x[:, 1]), max(x[:, 1]), x1.shape[0] )
  return x1, x2


# plot hyperplane and data points
def plot_points_and_line(x, y, b, w):
  plt.plot(x[y == 1, 0], x[y == 1, 1], 'go', label="pos")
  plt.plot(x[y == -1, 0], x[y == -1, 1], 'ro', label="neg")
  domain = np.linspace(min(x[:, 0]), max(x[:, 0]))
  xs, ys = get_hyperplane(domain, w, b, x)
  # if all weight vector coefficients excluding the bias were not zero
  if len(ys):
    plt.plot(xs, ys, color='b', linestyle='-')
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.legend(loc='upper left')
  plt.show()


# Main function
if __name__ == '__main__':
  x, y = pcp.generate_points(n=200, blob_centers=[(-1, -1), (1, 1)], cluster_std=0.5)
  w, b, _ = pcp.perceptron(x, y)
  plot_points_and_line(x, y, b, w)