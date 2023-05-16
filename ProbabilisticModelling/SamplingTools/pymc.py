'''
  PyMC example for a coin flip
'''

# libraries
import numpy as np
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('dark', {'axes.grid' : False})

# globals
RAND=33
CHAINS=2


# main
if __name__ == '__main__':

  np.random.seed(RAND)

  # coin-flip models
  with pm.Model() as coin_flip:
	  # data
    flips = np.random.randint(low=0, high=2, size=100)
	  # prior (coin-flip heads probability)
    theta = pm.Uniform('theta', lower=0, upper=1, transform=None)
	  # likelihood
    y = pm.Bernoulli('y', p=theta, observed=flips)
    trace = pm.sample(2000, tune=500, chains=CHAINS, progressbar=False, return_inferencedata=True)
    sampling_data = az.extract(trace)


  # plots
  samples = sampling_data['theta']
  chain_nums = sampling_data['chain']
  chains = [samples[chain_nums==n] for n in range(CHAINS)]
  colors = ['violet', 'aquamarine']

  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  for c in range(CHAINS):
    # prior samples distribution
    sns.kdeplot(data=chains[c], fill=True, ax=axs[0], color=colors[c], alpha=0.4)
    # sampling steps
    sns.lineplot(data=chains[c], ax=axs[1], color=colors[c], alpha=0.4)

  axs[0].set_facecolor('black')
  axs[0].set_xlabel('Coin-flip heads probability')
  axs[0].set_ylabel('Probability density')
  axs[0].set_title('Prior samples distribution')
  
  axs[1].set_facecolor('black')
  axs[1].set_xlabel('Sampling step #')
  axs[1].set_ylabel('Sampled value $\\theta$')
  axs[1].set_title('MCMC sampling of coin-flip heads probabilities')

  fig.suptitle(f'MCMC sampling with PyMC with {CHAINS} chains')
  plt.tight_layout()
  plt.show()
