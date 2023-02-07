# simulate data for logisitic regression and further analysis

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve


'''
  Logistic regression with gradient descent
  gradient = - (1/N) * summation (yn*xn)/(1 + e^(yn*w^t*xn))
  returns: final weight vector after 'iter' updates
'''
def logistic_regression(X, y, iter, eta=0.01, w=None):
  
  # stipulations
  N, d = X.shape
  if w is None:
    w = np.zeros(d)
  y.reshape(-1, 1)

  for _ in range(iter):
    gradient = (-1/N) * sum([(y[i] * X[i, :])/(1 + np.exp(y[i] * np.dot(w.T, X[i, :])) ) for i in range(N) ]) 
    w -= eta*gradient

  return w, loss_func(X, y, w)


'''
  Prediction function using probability 0.5 as threshold
  Function used: theta(w^T * X), where theta(x) = 1/(1 + e^(-x))
  Label data as 0 and 1 accordingly
  returns: ndarrays of probabilities and labels
'''
def predictions(X, w, threshold=0.5):
  y_hat, y_theta = [], []
  for xn in X:
    s = np.dot(w, xn)
    theta = 1/(1 + np.exp(-s))
    y_hat.append(1 if theta > threshold else 0)
    y_theta.append(theta)
  return y_hat, y_theta


# generate data from sklearn's make_blobs and return it split for training & testing
def generate_data(samples, features, centers, test_size=0.3, random_state=33):
  X, y = make_blobs(n_samples=samples, n_features=features, centers=centers, random_state=random_state)
  # adding column of 1's to X; can fit without the intercept as well
  X = np.vstack((np.ones(X.shape[0]), X.T)).T
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

  return X_train, X_test, y_train, y_test


# loss function
def loss_func(X, y, w):
  # calculating loss = (1/N) * summation ln(1 + e^(-yn*w^t*xn))
  return sum([np.log(1 + np.exp((-1) * y[i] * np.dot(w.T, X[i, :]))) for i in range(len(X))]) / len(X)


# helper for plot_points_and_line
def discriminant(x1, w):
  x2 = -x1 * w[1] / w[2] - w[0] / w[2]
  return x2

'''
  custom function to plot separator line and data points for logistic regression
  only for 2D data
'''
def plot_points_and_line(X, y, w):
  domain = np.linspace(X[:, 1].min(), X[:, 1].max())
  # showing the learned decision hyperplane and the data
  plt.plot(X[y == 1, 1], X[y == 1, 2], 'go', label="class 1")
  plt.plot(X[y == 0, 1], X[y == 0, 2], 'rx', label="class 0")
  plt.plot(domain, discriminant(domain, w), 'b-')
  plt.legend(loc="lower left")

  plt.title('Logistic regression decision boundary')
  plt.show()


# main function
if __name__ == '__main__':
  
  # generate data with 2 classes, not linearly separable
  X_train, X_test, y_train, y_test = generate_data(1000, 2, [(-1, -1), (1, 1)])

  # train for optimized weight vector
  w, loss = logistic_regression(X_train, y_train, 1000)
  print(f'''Logisitic regression:
    Final weight vector: {w}
    Final loss value: {loss}''')
  
  # probabilities for test set
  print('Testing ...............')
  y_hats, y_thetas = predictions(X_test, w)

  # metrics
  print(f'''Confusion Matrix:\n {confusion_matrix(y_test, y_hats)}
  Accuracy: {accuracy_score(y_test, y_hats):.2f}
  Recall: {recall_score(y_test, y_hats):.2f}
  Precision: {precision_score(y_test, y_hats):.2f}''')

  # AUC curve
  print(f'''Plotting AUC curve ........''')
  fpr, tpr, thresholds = roc_curve(y_test, y_thetas)
  auc = roc_auc_score(y_test, y_thetas)
  plt.plot([0, 1], [0, 1], '--', label='No Skill')
  plt.plot(fpr, tpr, 'go--', label=f'Logistic Regression, {auc=:.3f}')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc='lower right')
  plt.show()

  # Logistic regression plot
  print(f'''Plotting separator line from logistic regression for test data ........''')
  plot_points_and_line(X_test, y_test, w)