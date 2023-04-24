# Modelling differential equations final report for a bird spinning on a wheel

# libraries
import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


##################### update functions ##########################
'''
  ideal model: only gravity
  aa = g*sin(sa)/r
'''
def ideal(t, y):
  # variable values for current iteration
  sa, va = y

  g = 9.8      # gravitational acc (m/s^2)
  r = 0.6      # radius of wheel (m)

  # computing change in variables for current iteration
  dsadt = va
  dvadt = (g/r)*np.sin(sa)

  return dsadt, dvadt


'''
  air resistance model: air resistance + gravity
  aa = g*sin(sa)/r - k*r*va^2/m
'''
def air_res(t, y):
  sa, va = y

  g = 9.8
  r = 0.6
  k = 0.05  # air resistance coefficent (dimensionless)
  m = 3   # mass of cockatoo (kg)

  dsadt = va
  dvadt = (g/r)*np.sin(sa) - np.sign(va)*k*(va**2)*r/m

  return dsadt, dvadt


'''
  fluctuating radius model: air resistance + gravity + cockatoo moves
  aa = g*sin(sa)/(r+x*sin(sa)) - k*(r+x*sin(sa))*va^2/m
'''
def fluc_rad(t, y):
  sa, va = y

  g = 9.8
  x = 0.1  # length variation ability of cockatoo
  r = 0.6 + x*np.sin(sa)
  k = 0.05  # air resistance coefficent (dimensionless)
  m = 3   # mass of cockatoo (kg)

  dsadt = va
  dvadt = (g/r)*np.sin(sa) - np.sign(va)*k*(va**2)*r/m

  return dsadt, dvadt

###########################################################


# main
if __name__ == '__main__':
  
  # initial conditions
  sa0 = 0       # angular displacement: rad
  va0 = 2       # angular velocity: rad/s
  inits = [sa0, va0]

  # scope of analysis
  start, end = 0, 15
  dt = 0.05
  times = np.arange(start, end + dt, dt)

  # switch the model function to simulate various cases
  res = solve_ivp(fun=ideal, t_span=(start, end), y0=inits, t_eval=times)
  sa = res.y[0,:]
  va = res.y[1,:]
  t = res.t

  # plots
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))

  axs[0].grid(False)
  axs[0].set_facecolor('black')
  axs[0].plot(t, sa, 'b-', label='\u03B8 (rad)')
  axs[0].plot(t, va, 'r-', label='$\omega$ (rad $s^{-1}$)')
  axs[0].set_xlabel('Time (seconds)')
  axs[0].legend(loc='upper right')

  axs[1].grid(False)
  axs[1].set_facecolor('black')
  axs[1].plot(sa, va, 'm-')
  axs[1].set_xlabel('Angular displacement (rad)')
  axs[1].set_ylabel('Angular velocity (rad $s^{-1}$)')

  fig.suptitle('Fluctuating Radius Model Initial conditions: \u03B8 = ' + str(sa0) + ' rad, $\omega$ = ' + str(va0) + ' rad$s^{-1}$')
  plt.tight_layout()
  plt.savefig('plots.jpg')
  plt.show()