# plot contour for loss function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import widrow_hoff as wh
from sklearn import preprocessing


# plot contour for loss function in linear regression
def plot_contour (X, y):

  # meshgrid
  w0s, w1s = np.meshgrid(np.linspace(-20, 20, 201), np.linspace(-20, 20, 201))

  # plot weight vector coordinates with contour
  losses = np.array([ wh.loss_func(X, y, np.array([w0, w1]).reshape(-1, 1)) for w0, w1 in zip(np.ravel(w0s), np.ravel(w1s)) ])
  zs = losses.reshape(w0s.shape) # shape it to mesh shape
  wh_batch, _ = wh.widrow_hoff(X, y, 1000, True)
  wh_stoch, _ = wh.widrow_hoff(X, y, 10000, False)
  whb, whs = np.array(wh_batch), np.array(wh_stoch)

  plt.subplot(1, 2, 1) # divide plot box into [1, 2] grid and plot in #1
  plt.contour(w0s, w1s, zs, 100, cmap='jet')
  # plot weight vector components
  plt.plot(whb[:,0], whb[:,1], '--.r', label='Batch')
  plt.plot(whs[:,0], whs[:,1], '--.g', label='Stochastic', alpha=0.6)
  plt.xlabel('$w_1$')
  plt.ylabel('$w_2$')
  plt.title('Unscaled inputs')
  

  # scaled inputs
  X = preprocessing.scale(X) # normalize the data distribution
  losses = np.array([ wh.loss_func(X, y, np.array([w0, w1]).reshape(-1, 1)) for w0, w1 in zip(np.ravel(w0s), np.ravel(w1s)) ])
  zs = losses.reshape(w0s.shape)
  wh_batch, _ = wh.widrow_hoff(X, y, 1000, True)
  wh_stoch, _ = wh.widrow_hoff(X, y, 10000, False)
  whb, whs = np.array(wh_batch), np.array(wh_stoch)

  plt.subplot(1, 2, 2)
  plt.contour(w0s, w1s, zs, 100, cmap='jet')
  plt.plot(whb[:,0], whb[:,1], '--.r', label='Batch')
  plt.plot(whs[:,0], whs[:,1], '--.g', label='Stochastic', alpha=0.6)
  plt.xlabel('$w_1$')
  plt.ylabel('$w_2$')
  plt.title('Scaled inputs')

  plt.show()


# main function
if __name__ == '__main__':

  df = pd.read_csv('assignment2wine.csv')
  X, y = df[['density', 'residualSugar']], df[['alcohol']]
  plot_contour(np.array(X), np.array(y))
  