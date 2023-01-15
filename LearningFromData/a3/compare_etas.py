# compare effect on logistic regression model by varying learning rates

import matplotlib.pyplot as plt
import log_reg as lr
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve

'''
  train & test as in 'log_reg.py' for different etas
  params: X: independent variable matrix, y: dependent variable list, etas: list of etas
  returns: list of predicted probabilities
'''
def train_diff_etas(X_train, X_test, y_train, etas):
  # initial training
  fw, _ = lr.logistic_regression(X_train, y_train, 1000)
  # take only a part of data to further refine weight vector
  l = len(X_train) // 10
  X_train, y_train = X_train[:l], y_train[:l]
  data = np.empty((len(etas), X_test.shape[0]))
  for i, eta in enumerate(etas):
    w, _ = lr.logistic_regression(X_train, y_train, 1000, eta, fw)
    _, y_thetas = lr.predictions(X_test, w)
    data[i] = y_thetas
  return data


# main function
if __name__ == '__main__':
  
  # data
  X_train, X_test, y_train, y_test = lr.generate_data(samples=1000, features=2, centers=[(-1, -1), (1, 1)])
  # testing
  etas = [10**p for p in range(-4, 2)]
  data = train_diff_etas(X_train, X_test, y_train, etas)

  # plot AUC curves for all learning rates
  plt.figure(figsize=(10, 7))
  for d, eta in zip(data, etas):
    fpr, tpr, thresholds = roc_curve(y_test, d)
    auc = roc_auc_score(y_test, d)
    plt.plot(fpr, tpr, '--', label=f'\u03B7={eta}, {auc=:.3f}')

  plt.plot([0, 1], [0, 1], 'k-', label='No Skill')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc='lower right')
  plt.show()

# no conclusion, the AUC curves for all etas showed more or less the same performance