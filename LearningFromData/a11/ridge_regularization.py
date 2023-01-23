'''
  Analysis of Ridge regularization
  Weight vector w = (phi^T * phi + delta^2 * I)^-1 * phi^T * y
'''


# libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark', {'axes.grid' : False})

import sys
sys.path.append("..")
import a10.gamma_analysis as ga


# main
if __name__ == '__main__':
  X, y = ga.gen_data(10)
  
  y_hats = []
  gamma = 1
  deltas = [0.05, 0.1, 0.5, 1, 2.5, 5]
  xlo, xhi = np.min(X), np.max(X)
  xs = np.linspace(xlo, xhi, 400)

  for delta in deltas:
    phi = ga.gen_RBF_mat(X, gamma)
    w = np.linalg.inv((phi.T @ phi) + (np.eye(len(X))* delta**2)) @ phi.T @ y
    y_hats.append( ga.rbf_outputs(w, gamma, xs, X) )

  # plot graphs for the deltas
  fig, axs = plt.subplots(len(deltas)//2, 2, figsize=(12, 10))
  for i in range(len(deltas)):
    axs[i//2, i%2].plot(xs, y_hats[i], 'k-')
    axs[i//2, i%2].plot(X, y, 'go')
    axs[i//2, i%2].set_title(f'\u03B4={deltas[i]}')
  fig.suptitle('RBF for 1D data with ridge regularization')
  plt.tight_layout()
  plt.show()
