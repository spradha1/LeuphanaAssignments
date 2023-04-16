'''
  Simulating data from paper: http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf
'''


# libraries
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark', {'axes.grid' : False})


# global
CUSTOMERS=40
TIME_PERIODS=104
MAX_TR_RATE=12.0


# main
if __name__ == '__main__':
    
  # Poisson disribution of number of purchases by customers per time period
  ran_gen = np.random.default_rng()
  transaction_rates = np.random.random(size=CUSTOMERS)*MAX_TR_RATE
  customer_events_per_period = ran_gen.poisson(lam=transaction_rates, size=(TIME_PERIODS, CUSTOMERS)).T
  
  # randomly generating timestamps off of the purchase counts
  customers_timestamps = {}
  all_timestamps = np.array([])
  for i, cus in enumerate(customer_events_per_period):
    cus_timestamps = np.array([])
    for j, event_num in enumerate(cus):
      timestamps = np.sort(np.random.random_sample(size=event_num)) + j
      cus_timestamps = np.concatenate((cus_timestamps, timestamps))
    customers_timestamps[i] = cus_timestamps
    all_timestamps = np.concatenate((all_timestamps, cus_timestamps))

  # differences between all timestamps
  sorted_all_timestamps = np.sort(all_timestamps)
  timestamps_diffs = np.diff(sorted_all_timestamps)
  

  ######### visualize distributions #########
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))

  # Exponential distribution of differences in the timestamps of the transactions
  sns.kdeplot(data=timestamps_diffs, fill=True, ax=axs[0], color='darkseagreen')
  axs[0].set_facecolor('black')
  axs[0].set_xlabel('Time intervals between transactions')
  axs[0].set_ylabel('Probability density')
  axs[0].set_title('Exponential')
  axs[0].set_xlim(0)
  axs[0].margins(x=0)

  # Gamma distribution of transaction rates among customers
  sns.kdeplot(data=transaction_rates, fill=True, ax=axs[1], color='darkseagreen')
  axs[1].set_facecolor('black')
  axs[1].set_xlabel('Transaction rates among customers')
  axs[1].set_ylabel('Probability density')
  axs[1].set_title('Gamma')
  axs[1].set_xlim(0)
  axs[1].margins(x=0)

  fig.suptitle("Simulation of data from the paper, \"Counting Your Customers\" the Easy Way: An Alternative to the Pareto/NBD Model")
  plt.tight_layout()
  plt.show()