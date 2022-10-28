'''
  @author: Sanjiv Pradhanang
  Learning from Data: Assignment 1
  Linear equation plot
'''


import numpy as np  
import matplotlib.pyplot as plt 

'''
  @params: RHS of an equation with LHS evaluating to one variable, range of input values for variables on RHS
  plots an equation
'''
def drawLine(eqn, x_range):  
  x = np.array(x_range)  
  y = eqn(x)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.plot(x, y)  
  plt.show()  

# input equation
def input_equation(x):
  return (-2/3)*x - (1/3)


# Main function
if __name__ == '__main__':
  drawLine(input_equation, range(0, 10))