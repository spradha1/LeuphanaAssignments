# Using Widrow-Hoff Algorithm on wine data


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



"""
  Widrow-Hoff learning algorithmin batch and stochastic mode
  :param X: matrix of independent variables (numpy array)
  :param y: corresponding labels
  :param iter: number of iterations of gradient descent - default: 250
  :param eta: learning rate - default: 0.01
  :param batch: batch (True) or stochastic (False) gradient descent
  :return: final weights history, history of loss function
"""
def widrow_hoff(X, y, iter, batch, eta=1e-2):
  n = X.shape[0]
  w = np.zeros(X.shape[1]).reshape(-1, 1)     # Starting with zero weight vector
  w_history, loss_history = [w.flatten()], [loss_func(X, y, w)]

  for _ in range(iter):
    if batch:
      # batch mode
      y_hat = np.dot(X, w).reshape(-1, 1)
      v = np.dot(X.T, y_hat - y).reshape(-1, 1)
    else:
      # stochastic mode
      i = np.random.randint(0, n)
      Xi, yi = np.array([X[i]]), np.array([y[i]])
      y_hat = np.dot(Xi, w).reshape(-1, 1)
      v = np.dot(Xi.T, y_hat - yi).reshape(-1, 1)

    # gradient formula used: eta * X^T * (Xw - y) (normalized)
    v = v/np.linalg.norm(v)
    w -= v*eta
    w_history.append(w.flatten())
    loss_history.append(loss_func(X, y, w))

  return w_history, loss_history


# loss function for linear regression: (1/2) * ||Xw - y||^2
def loss_func(X, y, w):
  lv = np.dot(X, w) - y
  return float( 0.5 * np.dot(lv.T, lv) )


# Main function
if __name__ == '__main__':

  df = pd.read_csv('assignment2wine.csv')

  # training for predicting residualSugar
  X, y = df.loc[:, df.columns != 'residualSugar'], df[['residualSugar']]
  # adding intercept column
  X.insert(0, 'intercept', [1]*len(X))
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
  np_X_train, np_y_train = np.array(X_train), np.array(y_train)
  np_X_test, np_y_test = np.array(X_test), np.array(y_test)

  ws_b, losses_b = widrow_hoff(np_X_train, np_y_train, 25000, True, 0.02)
  ws_s, losses_s = widrow_hoff(np_X_train, np_y_train, 25000, False, 0.02)

  # Batch
  y_hat = np.dot(np_X_test, ws_b[-1])
  print(f'''Batch mode:
  Widrow-Hoff predicting residualSugar on training set:
    Final weight vector: {ws_b[-1]}
    Final loss value: {losses_b[-1]}
  MSE for test set: {mean_squared_error(np_y_test, y_hat):.2f}\n''')

  # Stochastic
  y_hat = np.dot(np_X_test, ws_s[-1])
  print(f'''Stochastic mode:
  Widrow-Hoff predicting residualSugar on training set:
    Final weight vector: {ws_s[-1]}
    Final loss value: {losses_s[-1]}
  MSE for test set: {mean_squared_error(np_y_test, y_hat):.2f}''')