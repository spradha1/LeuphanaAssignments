# analyze behavior of PLA on with change in different variables

import perceptron as pcp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark', {'axes.grid' : False})


def sample_sizes_to_mistakes():
  total_samples = 1000
  x, y = pcp.generate_points(n=total_samples)
  sample_sizes = range(20, total_samples+1, 20)
  mistakes = []
  for portion in sample_sizes:
    px, py = x[:portion+1], y[:portion+1]
    _, _, m = pcp.perceptron(px, py)
    mistakes.append(m)
  return sample_sizes, mistakes, 'Sample size'


def learning_rates_to_mistakes():
  x, y = pcp.generate_points(n=5000)
  sample_etas = range(1, 26)
  mistakes = []
  for e in sample_etas:
    _, _, m = pcp.perceptron(x, y, e)
    mistakes.append(m)
  return sample_etas, mistakes, 'Learning rate'
  

def maximal_margins_to_mistakes():
  x, y = pcp.generate_points(n=500, blob_centers=[(-5, -5), (-4.5, -4.5)])
  margins = np.arange(0.5, 10.1, 0.5)
  mistakes = []
  for m in margins:
    _, _, m = pcp.perceptron(x, y)
    x[y == 1] += 0.5
    mistakes.append(m)
  return margins, mistakes, 'Distance between centres'


def dimensions_to_mistakes():
  features = range(2, 13)
  mistakes = []
  for f in features:
    x, y = pcp.generate_points(n=500, d=f)
    _, _, m = pcp.perceptron(x, y)
    mistakes.append(m)
  return features, mistakes, 'Input dimensions'


def plot_all(vals):
  l = len(vals)
  fig, axs = plt.subplots(l//2, 2, figsize=(l*5//2, 8))
  for i in range(l):
    ax = axs[i//2, i%2]
    ax.plot(vals[i][0], vals[i][1], '.-')
    ax.set_xlabel(vals[i][2])
    ax.set_ylabel('Mistakes')
  
  fig.suptitle('Behaviour of PLA with change in variables')
  plt.tight_layout()
  plt.show()


# Main function
if __name__ == '__main__':
  res = []
  res.append(sample_sizes_to_mistakes())
  res.append(learning_rates_to_mistakes())
  res.append(maximal_margins_to_mistakes())
  res.append(dimensions_to_mistakes())

  plot_all(res)