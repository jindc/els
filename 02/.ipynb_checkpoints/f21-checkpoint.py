import numpy as np
from numpy import *
from numpy import linalg
from matplotlib import pyplot as plt
from IPython.display import display
import math


def get_k_nearest(X, Y, point_x):
  return []

def yfunc(sample_x):
  return np.power(np.e, -8 * np.power(linalg.norm(sample_x, ord=2, axis=1), 2))


def get_distance(trainset_num, trainset_size, dim_list, k=1, predict_point=[0]):
  ret = []
  for dim in dim_list:
    distances = []
    predict_point_x = np.zeros([1, dim])
    predict_point_y = 1.0

    for i in range(trainset_num):
      samples = np.random.rand(trainset_size, dim) * 2 - 1
      real_values = yfunc(samples)
      knearest = get_k_nearest(samples, real_values, predict_point_x)
      distances.append(knearest[-1])

    average_distance = math.fsum(distances) / len(distances)
    ret.append(average_distance)
  return ret


dim_list = range(1, 11)
distances = get_distance(10, 10, dim_list, k=1)

plt.xlabel("Dimension")
plt.ylabel("average distance for nearest neighbor")
plt.title('Distance to 1-NN vs. Dimension')
plt.plot(dim_list, distances, 'ro--', )
dim_list, distances