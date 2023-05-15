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
    trace = pm.sample(2000, tune=500, chains=CHAINS, progressbar=False, return_inferencedata=False)
  
  # plots
  varname = trace.varnames[0]
  samples = trace[varname]
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  # prior samples distribution
  sns.kdeplot(data=samples, fill=True, ax=axs[0], color='darkseagreen')
  axs[0].set_facecolor('black')
  axs[0].set_xlabel('Coin-flip heads probability')
  axs[0].set_ylabel('Probability density')
  axs[0].set_title('Prior samples distribution')
  # sampling steps
  sns.lineplot(data=samples, ax=axs[1], color='darkseagreen')
  axs[1].set_facecolor('black')
  axs[1].set_xlabel('Sampling step #')
  axs[1].set_ylabel('Sampled value $\\theta$')
  axs[1].set_title('MCMC sampling of coin-flip heads probabilities')

  fig.suptitle("MCMC sampling with PyMC")
  plt.tight_layout()
  # plot chain distributions separately with arviz, setting return_inferencedata=True in sampler
  # az.plot_trace(trace, combined=True)
  plt.show()
