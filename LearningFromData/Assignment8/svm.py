# SVM implementation of quadratic programming

# libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cvxopt import matrix, solvers

sns.set_style('dark')


# target function: f(x2) = sign(x2)
def target_func(x2):
  return np.sign(x2)


'''
  generate data points
  (x1, x2): x1 in [0, 1), x2 in [-1, 1)
  n: # of points
'''
def generate_data(n):
  x1s = np.random.uniform(0.0, 1.0, n)
  x2s = np.random.uniform(-1.0, 1.0, n)
  return np.stack((x1s, x2s), axis=-1)


'''
  Quadratic programming solver
  Minimizing: (1/2)(x^T)*Q*x + (p^T)*x
  Constraints: inequality Gx<=h, equality Ax=b
  n = # of data points
'''
def qp_solver(Q, p, G, h, A, b, n):
  Q, p, G, h, A, b = matrix(Q), matrix(p), matrix(G), matrix(h), matrix(A, (1, n)), matrix(b)
  return solvers.qp(Q, p, G, h, A, b)


'''
  calculate weight vector and threshold from alphas, xs and ys
  w = summation alphan*xn*yn
  yn*((w^T)*x + thres) = 1
'''
def calc_w_b(alphas, xs, ys):
  w = np.zeros(len(xs[0]))
  for a, x, y in zip(alphas, xs, ys):
    w += a*y*x
  sv_ind = np.argmax(alphas)
  thres = ys[sv_ind] - np.dot(w, xs[sv_ind, :])
  return w, thres


# plot points, margin & hyperplane
def plot_it(xs, ys, w, thres):

  # input points
  plt.plot(xs[:, 0][ys == 1], xs[:, 1][ys == 1], 'go')
  plt.plot(xs[:, 0][ys == -1], xs[:, 1][ys == -1], 'ro')

  # points on the hyperplane: (w^T)*x + b = 0 => x2 = -(w1/w2)x1 - b/w2
  x1s, x2s = [0, 1], []
  x2s = np.array([-x1*w[0]/w[1] - thres/w[1] for x1 in x1s])
  margin = 1/np.linalg.norm(w)
  plt.plot(x1s, x2s, 'k-')
  plt.plot(x1s, x2s + margin, color='b', linestyle='dotted')
  plt.plot(x1s, x2s - margin, color='b', linestyle='dotted')
  plt.fill_between(x1s, x2s + margin, x2s - margin, alpha=0.3, color='c')

  plt.title('Support Vector Machine 2D')
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.show()


# main
if __name__ == '__main__':
  
  np.random.seed(33)
  n = 20 # of data points
  xs = np.array(generate_data(n))
  ys = np.array([target_func(x) for _, x in xs]).astype('float')
  
  # creating arrays for cvxopt solver
  Q = np.empty((n, n))
  for i, xiyi in enumerate(zip(xs, ys)):
    xi, yi = xiyi
    for j, xjyj in enumerate(zip(xs, ys)):
      xj, yj = xjyj
      Q[i][j] = yi*yj*np.dot(xi.T, xj)
  
  p = -1*np.ones(n)
  G = -1*np.eye(n)
  h = np.zeros(n)
  A = np.array(ys)
  b = [0.0]
  
  alphas = np.array(qp_solver(Q, p, G, h, A, b, n)['x']).flatten()
  print(f'{alphas=}')

  w, thres = calc_w_b(alphas, xs, ys)
  print(f'{w=}\n{thres=}')
  plot_it(xs, ys, w, thres)
