# compare effect on logistic regression model by varying learning rates

import matplotlib.pyplot as plt
import log_reg as lr
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve

'''
  train & test as in 'log_reg.py' for different etas
  params: X: independent variable matrix, y: dependent variable list, etas: list of etas
  returns: list of predicted probabilities and final weights
'''
def train_diff_etas(X_train, X_test, y_train, etas):
  # initial training
  fw, _ = lr.logistic_regression(X_train, y_train, 1000)
  # take only a part of data to further refine weight vector
  l = len(X_train) // 10
  X_train, y_train = X_train[:l], y_train[:l]
  data = np.empty((len(etas), X_test.shape[0]))
  weights = np.empty((len(etas), X_test.shape[1]))
  for i, eta in enumerate(etas):
    w, _ = lr.logistic_regression(X_train, y_train, 1000, eta, fw)
    _, y_thetas = lr.predictions(X_test, w)
    data[i] = y_thetas
    weights[i] = w
  return data, weights


# main function
if __name__ == '__main__':
  
  # data
  X_train, X_test, y_train, y_test = lr.generate_data(samples=1000, features=2, centers=[(-1, -1), (1, 1)])
  # testing
  etas = [10**p for p in range(-4, 2)]
  data, weights = train_diff_etas(X_train, X_test, y_train, etas)

  # plot AUC curves for all learning rates
  plt.figure(figsize=(10, 7))
  for d, eta in zip(data, etas):
    fpr, tpr, thresholds = roc_curve(y_test, d)
    auc = roc_auc_score(y_test, d)
    plt.plot(fpr, tpr, '--', label=f'$\eta$={eta}, {auc=:.3f}')

  plt.plot([0, 1], [0, 1], 'k-', label='No Skill')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc='lower right')
  plt.show()

  # plot separator lines for all learning rates
  fig, axs = plt.subplots(len(etas)//2, 2, figsize=(10, 10))
  for i in range(len(etas)):
    ax = axs[i//2, i%2]
    domain = np.linspace(X_test[:, 1].min(), X_test[:, 1].max())
    ax.plot(X_test[y_test == 1, 1], X_test[y_test == 1, 2], 'go', label="class 1")
    ax.plot(X_test[y_test == 0, 1], X_test[y_test == 0, 2], 'rx', label="class 0")
    ax.plot(domain, lr.discriminant(domain, weights[i]), 'b-')
    ax.set_title(f'$\eta$={etas[i]}')
    ax.legend(loc="lower left")

  fig.suptitle('Logistic regression decision boundary')
  plt.tight_layout()
  plt.show()

# no changes from extra training of model, the classification performance seems to be the same