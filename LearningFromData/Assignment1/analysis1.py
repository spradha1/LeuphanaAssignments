# plot labelled points with separator

import perceptron as pcp
import numpy as np
import matplotlib.pyplot as plt


def plot_hyperplane(x1, w, b, x):
  # equation of the hyperplane - from f(x) formula
  if w[1] != 0:
    x2 = -x1*w[0]/w[1] - b/w[1]
  elif w[0] != 0:
    x1 = np.full(x1.shape, -b/w[0])
    x2 = np.linspace( min(x[:, 1]), max(x[:, 1]), x1.shape[0] )
  return x1, x2

def plot_points_and_line(x, y, b, w):
  # showing the learned decision hyperplane and the data
  plt.plot(x[y == 1, 0], x[y == 1, 1], 'go', label="pos")
  plt.plot(x[y == -1, 0], x[y == -1, 1], 'ro', label="neg")
  domain = np.linspace(min(x[:, 0]), max(x[:, 0]))
  xs, ys = plot_hyperplane(domain, w, b, x)
  plt.plot(xs, ys, color='b', linestyle='-')
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.legend(loc='upper left')
  plt.show()

# Main function
if __name__ == '__main__':
  x, y = pcp.generate_points(n=200, blob_centers=[(-1, -1), (1, 1)], cluster_std=0.5)
  w, b, _, _ = pcp.perceptron(x, y)
  plot_points_and_line(x, y, b, w)