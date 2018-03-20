import numpy as np
from numpy import *
from numpy import linalg
from matplotlib import pyplot as plt
from IPython.display import display
import math


#computer mse for model
def get_mse(trainset_num, trainset_size, dim_list,x_start=-1,x_end=1,sample_function=None, estimator = None, real_point_y = 0.5,seed=None):
  mse_list = []
  variance_list = []
  sqbias_list = []
  predict_rlist =[]
  if seed != None:
    np.random.seed(seed)
  samples_map = {}
  for dim in dim_list:
    for i in range(trainset_num):
      samples = np.random.rand(trainset_size, dim) * (x_end - x_start)+x_start
      samples_map[(dim,i)] = samples

  for dim in dim_list:
    predict_list = []
    predict_point_x = np.zeros([1, dim])
    for i in range(trainset_num):
      real_values = sample_function(samples_map[(dim,i)])
      est = estimator()
      est.fit(samples_map[(dim,i)],real_values)
      pval = est.predict(predict_point_x)
      predict_list.append(pval[0])

    predict_rlist.append(math.fsum( predict_list )/len(predict_list))

    average_mse = math.fsum( math.pow(real_point_y - v ,2) for v in predict_list )/len(predict_list)
    mse_list.append(average_mse)

    predict_avg = math.fsum(predict_list)/len(predict_list)
    average_variance = math.fsum(math.pow( v-predict_avg, 2) for v in predict_list) / len(predict_list)
    variance_list.append(average_variance)

    average_sqbias = math.fsum(math.pow(real_point_y - predict_avg, 2) for v in predict_list) / len(predict_list)
    sqbias_list.append(average_sqbias)
  return mse_list,variance_list,sqbias_list,predict_rlist


def draw_mse(dim_list, mse, variance, sqbias=None, predict_list=None):
  plt.xlabel("Dimension")
  plt.ylabel("Mse")
  plt.title('MSE vs. Dimension')

  plt.plot(dim_list, mse, 'ro--', label="MSE")
  plt.plot(dim_list, variance, 'go--', label="Variance")

  if sqbias != None:
    plt.plot(dim_list, sqbias, 'bo--', label="Sq.Bias")
  if predict_list != None:
    plt.plot(dim_list, predict_list, 'yo--', label="predict value")
  plt.legend(loc="upper left")

  plt.show()