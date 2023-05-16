'''
  Python JAGS example for a coin flip
'''

# libraries
import numpy as np
import pyjags
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('dark', {'axes.grid' : False})

# globals
RAND=33
CHAINS=2
DATA_SIZE=100

# main
if __name__ == '__main__':

  np.random.seed(RAND)

  # data
  y = np.random.randint(low=0, high=2, size=DATA_SIZE)
  # jags stuff
  jags_data = {"y": list(y), "N": DATA_SIZE}
  parameters = ['theta']
  jags_model_string = '''
    model {
      theta ~ dunif(0, 1)
      for (i in 1:N) {
        y[i] ~ dbern(theta)
      }
    }
  '''
  jags_model = pyjags.Model(code=jags_model_string, data=jags_data, chains=CHAINS)
  jags_samples = jags_model.sample(iterations=2000, vars=parameters)
  
  # plots
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  sample_chains = jags_samples['theta'][0]
  colors = ['orangered', 'green']

  for c in range(CHAINS):
    # prior samples distribution
    sns.kdeplot(data=sample_chains[:, c], ax=axs[0], color=colors[c], fill=True, alpha=0.4)
    # sampling steps
    sns.lineplot(data=sample_chains[:, c], ax=axs[1], color=colors[c], alpha=0.4)

  axs[0].set_facecolor('black')
  axs[0].set_xlabel('Coin-flip heads probability')
  axs[0].set_ylabel('Probability density')
  axs[0].set_title('Prior samples distribution')
    
  axs[1].set_facecolor('black')
  axs[1].set_xlabel('Sampling step #')
  axs[1].set_ylabel('Sampled value $\\theta$')
  axs[1].set_title('MCMC sampling of coin-flip heads probabilities')

  fig.suptitle(f'MCMC sampling with pyjags with {CHAINS} chains')
  plt.tight_layout()
  plt.show()
