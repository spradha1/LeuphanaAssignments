# Modelling simple differential equation with initial value condition

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp

sns.set()


# variable update function
def updater(t, y):

  bkf = 12 # brakes
  dgf = 0.3 # drag
  
  # variable values for current iteration
  s, v = y
  
  # computing change in variables for current iteration
  dsdt = v
  dvdt = (-1)*(bkf + dgf*v) if v > 0 else 0

  return dsdt, dvdt


# main function
if __name__ == '__main__':
  
  s0 = 0 # initial distance travelled from point of braking
  v0 = 75 # initial speed m/s
  ivs = [s0, v0]

  start, end = 0, 5 # time span
  dt = 0.25
  times = np.arange(start, end + dt, dt)

  pl_dist = 100 # distance to pit lane at braking
  pl_spd = 20 # speed limit in pit lane

  #### solve_ivp ####
  res = solve_ivp(fun=updater, t_span=(start, end), y0=ivs, t_eval=times)
  s = res.y[0,:]
  v = res.y[1,:]
  t = res.t
  print(times, t)
  
  # plot s vs. t
  fig, axs = plt.subplots()
  axs.grid(False)
  axs.set_facecolor('black')
  axs.plot(t, s, 'bo-')
  axs.set_xlabel('Time (s)')
  axs.set_ylabel('Distance travelled (m)')
  axs.set_title('Deceleration of a racecar from point of braking')
  plt.show()

  # plot v vs. t
  fig, axs = plt.subplots()
  axs.grid(False)
  axs.set_facecolor('black')
  axs.plot(t, v, 'bo-')
  axs.set_xlabel('Time (s)')
  axs.set_ylabel('Velocity (m/s)')
  axs.set_title('Deceleration of a racecar from point of braking')
  plt.show()

  # plot v vs. s
  fig, axs = plt.subplots()
  axs.grid(False)
  axs.set_facecolor('black')
  axs.plot(s, v, 'bo-')
  axs.plot([pl_dist, pl_dist], [v[0], v[-1]], 'r--', label='Pit lane entry')
  axs.plot([0, max(s[-1], pl_dist + 10)], [pl_spd, pl_spd], 'g--', label='Speed limit in pit lane')
  axs.set_xlabel('Distance travelled (m)')
  axs.set_ylabel('Velocity (m/s)')
  axs.set_title('Deceleration of a racecar from point of braking')
  plt.legend()
  plt.show()
