import numpy as np
from numpy import *
from numpy import linalg
from matplotlib import pyplot as plt
from IPython.display import display
import math


def get_k_nearest(X, Y, point_x):
  dlist = [(X[i],Y[i],np.linalg.norm(np.array(X[i])-np.array(point_x),axis=1).tolist()[0]) for i in range(len(X)) ]
  dlist.sort(key=lambda item:item[2])
  #print( dlist)
  return dlist[0]

def get_old(X, Y, point_x):
  point_x = np.hstack((np.array(point_x),np.array([[1]])))
  samples_x_mat = np.mat(np.hstack( (np.array(X),np.ones([np.array(X).shape[0],1]))))
  samples_y_mat = np.mat(Y).reshape([-1,1])
  beta = (samples_x_mat.T * samples_x_mat).I * samples_x_mat.T * samples_y_mat
  ret =  (None,np.mat(point_x) * beta,None)
  return ret


def yfunc(sample_x):
  return np.power(np.e, -8 * np.power(linalg.norm(sample_x, ord=2, axis=1), 2))

def one_pow_func(samples_x):
  ret = [ math.pow(math.e, -8 *  math.pow(v[0],2)) for v in samples_x]
  return ret

def trinor(sample_x):
  return [ 0.5 * math.pow(v[0]+1,3) for v in sample_x ]

def get_distance(trainset_num, trainset_size, dim_list, yfunction=None,k=1, predict_point=[0]):
  ret = []
  for dim in dim_list:
    distances = []
    predict_point_x = np.zeros([1, dim])
    predict_point_y = 1.0

    for i in range(trainset_num):
      samples = np.random.rand(trainset_size, dim) * 2 - 1
      real_values = yfunction(samples)
      knearest = get_k_nearest(samples, real_values, predict_point_x)
      distances.append(knearest[-1])

    average_distance = math.fsum(distances) / len(distances)
    ret.append(average_distance)
  return ret

def draw_distance():
  dim_list = range(1,11)
  distances = get_distance(50, 1000, dim_list, k=1,yfunction=yfunc)

  plt.xlabel("Dimension")
  plt.ylabel("average distance for nearest neighbor")
  plt.title('Distance to 1-NN vs. Dimension')
  plt.plot(dim_list, distances, 'ro--', )
  dim_list, distances
  plt.show()


def get_gap(trainset_num, trainset_size, dim_list,  yfunction=None,k=1, predict_point=[0],real_point_y = 0.5):
  mse_list = []
  variance_list = []
  sqbias_list = []
  predict_rlist =[]
  distance_list =[]

  for dim in dim_list:
    predict_list = []
    predict_point_x = np.zeros([1, dim])
    real_point_y = 0.5
    distances = []
    for i in range(trainset_num):
      samples = np.random.rand(trainset_size, dim) * 2 - 1
      real_values = yfunction(samples)
      knearest = get_k_nearest(samples, real_values, predict_point_x)
      predict_list.append(knearest[1])
      distances.append(knearest[-1])

    predict_rlist.append(math.fsum( predict_list )/len(predict_list))
    distance_list.append(math.fsum(distances)/len(distances))

    average_mse = math.fsum( math.pow(real_point_y - v ,2) for v in predict_list )/len(predict_list)
    mse_list.append(average_mse)

    predict_avg = math.fsum(predict_list)/len(predict_list)
    average_variance = math.fsum(math.pow( v-predict_avg, 2) for v in predict_list) / len(predict_list)
    variance_list.append(average_variance)

    average_sqbias = math.fsum(math.pow(real_point_y - predict_avg, 2) for v in predict_list) / len(predict_list)
    sqbias_list.append(average_sqbias)

  return mse_list,variance_list,sqbias_list,predict_rlist,distance_list

def draw_mse():
  dim_list = range(1,11)
  mse, variance, sqbias,predict_list,distance_list = get_gap(100, 500, dim_list, k=1,yfunction=trinor,real_point_y=0.5)

  plt.xlabel("Dimension")
  plt.ylabel("Mse")
  plt.title('MSE vs. Dimension')

  plt.plot(dim_list, mse, 'ro--', label="MSE")
  plt.plot(dim_list,variance,'go--',label="Variance")
  plt.plot(dim_list,sqbias,'bo--',label="Sq.Bias")
  plt.plot(dim_list, predict_list, 'yo--', label="predict value")
  plt.plot(dim_list, distance_list, 'y*--', label="distance ")
  plt.legend(loc="upper left")

  plt.show()

def linear_undeter(sample_x):
  return [ v[0] + np.random.normal() for v in sample_x  ]

def cubic_undeter(sample_x):
  return [0.5 * math.pow(v[0] + 1, 3) + np.random.normal() for v in sample_x]

def get_mse(trainset_num, trainset_size, dim_list,  yfunction=None, estimator = get_k_nearest, real_point_y = 0.5):
  mse_list = []
  variance_list = []
  sqbias_list = []
  predict_rlist =[]

  for dim in dim_list:
    predict_list = []
    predict_point_x = np.zeros([1, dim])
    for i in range(trainset_num):
      samples = np.random.rand(trainset_size, dim) * 2 - 1
      real_values = yfunction(samples)
      knearest = estimator(samples, real_values, predict_point_x)
      predict_list.append(knearest[1])

    predict_rlist.append(math.fsum( predict_list )/len(predict_list))

    average_mse = math.fsum( math.pow(real_point_y - v ,2) for v in predict_list )/len(predict_list)
    mse_list.append(average_mse)

    predict_avg = math.fsum(predict_list)/len(predict_list)
    average_variance = math.fsum(math.pow( v-predict_avg, 2) for v in predict_list) / len(predict_list)
    variance_list.append(average_variance)

    average_sqbias = math.fsum(math.pow(real_point_y - predict_avg, 2) for v in predict_list) / len(predict_list)
    sqbias_list.append(average_sqbias)

  return mse_list,variance_list,sqbias_list,predict_rlist

def draw_29():
  dim_list = range(1, 11)

  y0_num = 5
  def get_epe(sample_function,estimator,yv):
    mse_list = []
    mse2_list = []
    for i in range(y0_num):
      y0 = yv + np.random.normal()
      mse_array = get_mse(50, 500, dim_list,estimator= estimator, yfunction=sample_function,real_point_y=y0)
      mse_list.append(mse_array[0])
    ret = np.average(np.array(mse_list),axis=0)
    return ret

  #linear
  mse_k = get_epe(linear_undeter,get_k_nearest,0)
  mse_linear = get_epe(linear_undeter,get_old,0)
  #cubic
  #mse_cubic = get_epe(cubic_undeter,0.5)

  plt.xlabel("Dimension")
  plt.ylabel("Mse")
  plt.title('MSE vs. Dimension')

  plt.plot(dim_list, mse_k, 'ro--', label="knearest MSE")
  plt.plot(dim_list, mse_linear, 'go--', label="linear  MSE")
  plt.plot(dim_list, mse_k/mse_linear, 'bo--', label="knearest/linear  MSE")
  # plt.plot(dim_list, variance, 'go--', label="Variance")
  # plt.plot(dim_list, sqbias, 'bo--', label="Sq.Bias")
  # plt.plot(dim_list, predict_list, 'yo--', label="predict value")
  # plt.plot(dim_list, distance_list, 'y*--', label="distance ")
  plt.legend(loc="upper left")
  plt.show()

#draw_29()
#draw_distance()
draw_mse()


