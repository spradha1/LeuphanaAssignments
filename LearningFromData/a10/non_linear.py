'''
  hyperplanes in 2D created by feature transform z = phi(x) = (1, x1^2, x2^2)
  h = sign(w^T * phi(x))
'''

# libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark", {'axes.grid' : False})


'''
  plot lines in 2D with feature transform hyperplane
  params: weight vector, horizontal bounds of the graph (x1, x2)
'''
def plot_boundary(w, xr=(-2, 2)):
  b, w1, w2 = w
  xlo, xhi = xr
  x1s = np.linspace(start=xlo, stop=xhi, num=np.abs(xhi-xlo)*40 + 1)

  # plot only points that are valid
  valid_x1s = np.array([])
  # two sets of points for above and below x2 axis
  pos_x2s, neg_x2s = np.array([]), np.array([])

  for x1 in x1s:
    val = -(w1/w2)*(x1**2) - b/w2
    if val >= 0:
      root = np.sqrt(val)
      valid_x1s = np.insert(valid_x1s, 0, x1)
      pos_x2s = np.insert(pos_x2s, 0, root)
      neg_x2s = np.insert(neg_x2s, 0, -root)
  
  plt.plot(valid_x1s, pos_x2s, color='b', linestyle='dotted')
  plt.plot(valid_x1s, neg_x2s, color='b', linestyle='dotted')
  plt.xlabel('x1')
  plt.xlabel('x2')
  plt.title(f'Boundary for \u03C6(x) = (1, x1^2, x2^2) with {w=}')
  plt.xlim(xlo, xhi)
  plt.ylim(xlo, xhi)
  plt.axis('equal')
  plt.tight_layout()
  plt.show()
      

# main
if __name__ == "__main__":
  # input weight vectors
  ws = [
    [1, -1, -1],
    [-1, 1, 1],
    [1, -1, -2],
    [1, 1, -1]
  ]
  for w in ws:
    plot_boundary(w)