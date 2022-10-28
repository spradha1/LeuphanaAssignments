# Perceptron learning algorithm assignment


import numpy as np
import math
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from sklearn.datasets import make_blobs


# perceptron learning algorithm for 2D points
def perceptron(x, y, eta=1):
  k = 0
  wk, bk = np.zeros(2), 0
  # learning rate
  R, iters = 0, 0
  for comps in x:
    R = max(R, math.sqrt( sum([math.pow(comp, 2) for comp in comps]) ) )
  errors = True

  # learning iterations
  while (errors):
    errors = False
    for i in range(len(x)):
      if y[i]*(np.dot(wk, x[i]) + bk) <= 0:
        wk += eta*y[i]*x[i]
        bk += eta*y[i]*math.pow(R, 2)
        k += 1
        errors = True
    iters += 1
  
  return [wk, bk, iters, k]


# check if data is linearly separable
def is_linearly_separable(pos_class, neg_class):
  pos_hull = ConvexHull(pos_class)
  neg_hull = ConvexHull(neg_class)
  return not Polygon(pos_hull.points).intersects(Polygon(neg_hull.points))


# generate 2D points with desired features
def generate_points(n=100, blob_centers=[], cluster_std=1):
  if len(blob_centers) == 0:
    x, y = make_blobs(n_samples=n, n_features=2, centers=2, random_state=1)
    if is_linearly_separable(x[y == 1], x[y == 0]):
      y[y == 0] = -1  # changing the label from 0 to -1, important for PLA
      return x, y
    else:
      return generate_points(n, blob_centers, cluster_std - 0.1)
  else:
    x, y = make_blobs(n_samples=n, n_features=2, centers=blob_centers, random_state=1, cluster_std=cluster_std)
    if is_linearly_separable(x[y == 1], x[y == 0]):
      y[y == 0] = -1
      return x, y 
    else:
      return generate_points(n, blob_centers, cluster_std - 0.1)
