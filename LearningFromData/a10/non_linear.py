'''
  hyperplanes in 2D created by feature transform z = phi(x) = (1, x1^2, x2^2)
  h = sign(w^T * phi(x))
'''

# libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark", {'axes.grid' : False})


# binary classifier 
def classify_2d(xs, w):
  b, w1, w2 = w
  return np.array([1 if w1*(x1**2) + w2*(x2**2) + b > 0 else 0 for x1, x2 in xs])

'''
  plot colors in 2D by their labels
  params: weight vector, horizontal bounds of the graph (x1, x2)
'''
def plot_canvas(ws, xr=(-2, 2)):
  xlo, xhi = xr
  xs = np.linspace(xlo, xhi, 250)
  canvas = np.array(np.meshgrid(xs, xs)).T.reshape(-1, 2)

  fig, axs = plt.subplots(len(ws)//2, 2, figsize=(10, 8))
  for (i, w) in enumerate(ws):
    labels = classify_2d(canvas, w)
    axc = axs[i//2, i%2]
    axc.plot(canvas[labels==1, 0], canvas[labels==1, 1], 'm.', label='+ve')
    axc.plot(canvas[labels==0, 0], canvas[labels==0, 1], 'k.', label='-ve')
    axc.set_title(f'{w=}')
    axc.set_aspect('equal')
    axc.set_xlim(xlo, xhi)
    axc.set_ylim(xlo, xhi)
    axc.legend(loc='upper left')
  fig.suptitle('Classification for 2D non-linear transforms $\phi$(x) = (1, x1^2, x2^2)')
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

  # plot color mesh for classified areas
  plot_canvas(ws)
  