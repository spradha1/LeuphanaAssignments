# analyze behavior of PLA on variation of inputs 

import perceptron as pcp
import numpy as np
import matplotlib.pyplot as plt


def sample_sizes_to_mistakes():
  total_samples = 1000
  x, y = pcp.generate_points(n=total_samples)
  sample_sizes = range(20, total_samples+1, 20)
  mistakes = []
  for portion in sample_sizes:
    px, py = x[:portion+1], y[:portion+1]
    _, _, m = pcp.perceptron(px, py)
    mistakes.append(m)
  plt.plot(sample_sizes, mistakes, '.-')
  plt.xlabel('Sample size')
  plt.ylabel('Mistakes')
  plt.show()


def learning_rates_to_mistakes():
  x, y = pcp.generate_points(n=5000)
  sample_etas = range(1, 26)
  mistakes = []
  for e in sample_etas:
    _, _, m = pcp.perceptron(x, y, e)
    mistakes.append(m)
  plt.plot(sample_etas, mistakes, '.-')
  plt.xlabel('Learning rate')
  plt.ylabel('Mistakes')
  plt.show()

  # note: if weights and threshold set to zero, learning rate has no influence
  

def maximal_margins_to_mistakes():
  x, y = pcp.generate_points(n=500, blob_centers=[(-5, -5), (-4.5, -4.5)])
  margins = np.arange(0.5, 10.1, 0.5)
  mistakes = []
  for m in margins:
    _, _, m = pcp.perceptron(x, y)
    x[y == 1] += 0.5
    mistakes.append(m)
  plt.plot(margins, mistakes, '.-')
  plt.xlabel('Distance between centres')
  plt.ylabel('Mistakes')
  plt.show()


# Main function
if __name__ == '__main__':
  sample_sizes_to_mistakes()
  learning_rates_to_mistakes()
  maximal_margins_to_mistakes()