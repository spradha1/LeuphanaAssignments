# Use perceptron from manually provided small dataset

import perceptron as pcp
import numpy as np


# Main function
if __name__ == '__main__':
  
  # data
  x = np.array([
    [1, 2], [3, 2], [2, 1], [3, 3]
  ])
  y = np.array([-1, 1, -1, 1])

  w, b, iters, k = pcp.perceptron(x, y)
  print(f'Iterations: {iters}\nWeight corrections: {k}\nFinal weight vecotr: {w}\nFinal threshold: {b:.2f}')
