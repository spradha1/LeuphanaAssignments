# Analyzing VC bound for positive intervals

# libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


'''
  Calculate VC bound for positive intervals
  Growth function mH(N) -> (1/2)*N^2 + (1/2)*N + 1
  VC bound -> 4*mH(2N)*e^(-(N/8)*ep^2)
  params: sample size, generalization tolerance
'''
def calc_vc_bound(N, ep=0.1):
  # growth function gives the maximum number of unique hypothesis/dichotomies possible
  max_dichotomies = 2*np.power(N, 2) + N + 1
  # RHS of modified Hoeffding's inequality
  delta_bound = 4*max_dichotomies*np.exp(-(N/8)*np.power(ep, 2))
  return delta_bound


# main
if __name__ == '__main__':
  
  # sample size inputs
  Ns = np.arange(0, 20001, 500)
  bounds = np.log10([calc_vc_bound(N) for N in Ns])

  # plot N vs. VC bound
  fig, ax = plt.subplots()
  ax.plot(Ns, bounds, c='#A6009A', ls='-')
  ax.plot([Ns[0], Ns[-1]], [0, 0], 'g-')
  ax.set_xlabel('Sample size')
  ax.set_ylabel('VC bounds ($log_{10}$)')
  ax.set_title('N & the VC dimension')
  ax.grid(False)
  ax.set_facecolor('black')
  plt.tight_layout()
  plt.show()