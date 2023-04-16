'''
  PageRank algorithm
'''

# libraries
import numpy as np


'''
  PageRank function
  params:
    M: input matrix (jth column contains i probabilities from j to i)
    iter: # of iterations
    damp: damoping factor
    atol: absolute tolerance for iteration
  returns:
    the final ranks, # of iterations taken, converged or not
'''
def pagerank(M, iter=100, damp=0.85, atol=0.01):
  n = M.shape[0]
  converged = False

  # every page starts with equal ranks
  scores = np.ones(n)/n

  # damped probabilities are distributed among all pages 
  damp_M = damp*M + (1 - damp)/n

  tmp_scores = scores
  for i in range(iter):
    scores = damp_M @ scores

    # stop iterating if ranks converged
    if np.allclose(scores, tmp_scores, atol=atol):
      converged = True
      break

    tmp_scores = scores
    
  return scores, i+1, converged


# main
if __name__ == '__main__':

  # pagerank input
  M = np.array([
    [0, 0, 0, 0, 1],
    [0.5, 0, 0, 0, 0],
    [0.5, 0, 0, 0, 0],
    [0, 1, 0.5, 0, 0],
    [0, 0, 0.5, 1, 0]
  ])

  ranks, num, conv = pagerank(M, 12)
  print(f'''Final Ranks: {ranks}\n'''
    f'''# of iterations: {num}\n'''
    f'''Convergence: {"Yes" if conv else "No"}'''    
  )