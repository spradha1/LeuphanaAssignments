'''
  Analyse the behaviour of gamma for radial basis functions
  Fitting RBF to a regression problem
'''


# libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark', {'axes.grid' : False})

from sklearn.datasets import make_regression


# generate 1D data for simplicity
def gen_data(n):
  X, y = make_regression(n_samples=n, n_features=1, n_targets=1, random_state=33, noise=1.0)
  return X, y


# generate RBF model matrix from data points
def gen_RBF_mat(X, gamma):
  N = len(X)
  phi = np.empty((N, N))
  for i in range(N):
    for j in range(0, i+1):
      tmp = np.exp(-1 * gamma * np.abs(X[i] - X[j]) )
      phi[i, j], phi[j, i] = tmp, tmp
  return phi


'''
  get values for RBF hypothesis
  summation [ wn * exp(-gamma*||x-xn||^2) ]
  params: weight vector, gamma, bounds on x
'''
def rbf_outputs(w, gamma, xs, X):
  ys = []
  for x in xs:
    tmp = 0
    for wn, xn in zip(w, X):
      tmp += wn*np.exp(-1 * gamma * np.abs(x - xn) )
    ys.append(tmp)
  return ys


# main
if __name__ == '__main__':
  X, y = gen_data(10)
  
  y_hats = []
  gammas = [1, 10]
  xlo, xhi = np.min(X), np.max(X)

  # making sure we get the data points that we are fitting to
  xs = np.sort(np.concatenate((np.linspace(xlo, xhi, 25), X.flatten())))

  for gamma in gammas:
    phi = gen_RBF_mat(X, gamma)
    phi_inv = np.linalg.inv(phi)
    w = np.matmul(phi_inv, y)     # get weight vector via w = phi^-1 * y
    y_hats.append( rbf_outputs(w, gamma, xs, X) )

  # plot graphs for the gammas
  fig, axs = plt.subplots(1, 2, figsize=(12, 5))
  for i in range(len(gammas)):
    axs[i].plot(xs, y_hats[i], 'k-')
    axs[i].plot(X, y, 'go')
    axs[i].set_title(f'gamma={gammas[i]}')
  fig.suptitle('RBF for 1D data')
  plt.tight_layout()
  plt.show()