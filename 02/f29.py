import numpy as np
from numpy import *
from numpy import linalg
from matplotlib import pyplot as plt
from IPython.display import display
import math
from utils.linear import Linear
from utils.mse import get_mse,draw_mse
from utils import samplesUtil
from utils.knearest import Knearest

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


def linear_undeter(sample_x):
  return [ v[0] + np.random.normal() for v in sample_x  ]

def cubic_undeter(sample_x):
  return [0.5 * math.pow(v[0] + 1, 3) + np.random.normal() for v in sample_x]

def linear_img():
  sample_size = 10000
  X = np.random.rand(sample_size,1)*2 -1
  linear = Linear()
  Y = samplesUtil.generate_sample_with_gauss_dim0(X)
  Y_predict = linear.fit(X,Y).predict(X)

  plt.scatter(X,Y)
  plt.plot(X,Y_predict)
  print(linear.beta)
  plt.show()

def draw_27():
  sample_function, estimator = samplesUtil.radial,Knearest
  dim_list = range(1,11)
  mse, variance, sqbias,predict_list = get_mse(100, 1000, dim_list
                        , sample_function=sample_function,estimator= estimator,real_point_y=1)
  draw_mse(dim_list,mse, variance, sqbias,None)

def draw_28():
  sample_function, estimator = samplesUtil.cubic_with_dim0,Knearest
  dim_list = range(1,11)
  mse, variance, sqbias,predict_list = get_mse(100, 500, dim_list
                        , sample_function=sample_function,estimator= estimator,real_point_y=0.5)
  draw_mse(dim_list,mse, variance, sqbias,None)

def draw_undeter_linear():
  dim_list = range(1, 11)
  mse, variance, sqbias, predict_list = get_mse(50, 500, dim_list
                                                , sample_function=samplesUtil.generate_sample_with_gauss_dim0
                                                , estimator=Knearest,
                                                real_point_y=0.0,seed=9)
  draw_mse(dim_list, mse, variance, sqbias, predict_list)

def draw_epe_undeter():
  dim_list = range(1, 11)
  real_y_num = 50
  real_y_list = []
  knearest_mse_list = []
  knearest_variance_list = []
  knearest_predict_list=[]
  linear_mse_list=[]
  np.random.seed(None)
  real_y_list = []
  for i in range(real_y_num):
    print(i)
    np.random.seed(None)
    real_y = 0.5 + np.random.normal()
    real_y_list.append(real_y)
    mse, variance, sqbias, predict_list = get_mse(100, 500, dim_list
                                                , sample_function=samplesUtil.cubic_with_gauss_dim0
                                                , estimator=Knearest,
                                                real_point_y=real_y)
    knearest_mse_list.append(mse)
    knearest_variance_list.append(variance)
    knearest_predict_list.append(predict_list)
    #print (real_y,predict_list)

  knearest_mse_avg = np.average(knearest_mse_list,axis=0)
  real_variance = [np.var(real_y_list)] * len(dim_list)
  knearest_predict_var_avg = np.average(knearest_variance_list, axis=0)
  knearest_predict_bias_avg = np.square(np.average(knearest_predict_list, axis=0) - np.average(real_y_list))

  plt.plot(dim_list,knearest_mse_avg,'ro--',label="mse on undeterminister")
  plt.plot(dim_list,real_variance,'go--',label="real value var")
  plt.plot(dim_list, knearest_predict_var_avg,'bo--', label="predict  value var")
  plt.plot(dim_list, knearest_predict_bias_avg,'b*--', label="predict  value bias")

  plt.legend()

  #plt.show()
  print("mse",knearest_mse_avg)
  print("real",real_variance)
  print("predict var",knearest_predict_var_avg)
  print("predict bias", knearest_predict_bias_avg)

def draw_29():
  def get_mse_value(sample_fucntion, estimator, seed_v, real_v):
    mse, variance, sqbias, predict_list = get_mse(100, 1000, dim_list
                                                  , sample_function=sample_fucntion
                                                  , estimator=estimator,
                                                  real_point_y=real_v, seed=seed_v)
    return mse

  dim_list = range(1, 11)
  real_y_num = 300
  knearest_mse_list = []
  linear_mse_list=[]

  cubic_knearest_mse_list = []
  cubic_linear_mse_list = []

  np.random.seed(None)
  seed_list=[ np.random.randint(10000000) for i in range(real_y_num)]
  print (seed_list)


  for i in range(real_y_num):
    print(i)
    np.random.seed(None)
    real_y = 0.0 + np.random.normal()

    knearest_mse_list.append(get_mse_value(samplesUtil.generate_sample_with_gauss_dim0,Knearest,seed_list[i],real_y))
    linear_mse_list.append(get_mse_value(samplesUtil.generate_sample_with_gauss_dim0, Linear, seed_list[i], real_y))

    np.random.seed(None)
    real_y = 0.5 + np.random.normal()
    cubic_knearest_mse_list.append(get_mse_value(samplesUtil.cubic_with_gauss_dim0, Knearest, seed_list[i], real_y))
    cubic_linear_mse_list.append(get_mse_value(samplesUtil.cubic_with_gauss_dim0, Linear, seed_list[i], real_y))

    #print (real_y,predict_list)

  knearest_mse_avg = np.average(knearest_mse_list,axis=0)
  linear_mse_avg = np.average(linear_mse_list, axis=0)
  cubic_knearest_mse_avg = np.average(cubic_knearest_mse_list,axis=0)
  cubic_linear_mse_avg = np.average(cubic_linear_mse_list, axis=0)

  plt.plot(dim_list,knearest_mse_avg/linear_mse_avg,'ro--',label="linear knearest/old ")
  plt.plot(dim_list, cubic_knearest_mse_avg / cubic_linear_mse_avg, 'ro--', label="cubic knearest/old ")


  plt.legend()
  #plt.show()
  print("k mse",knearest_mse_avg)
  print("lin mse", linear_mse_avg)
  print("k/lin mse", knearest_mse_avg/linear_mse_avg)

  print("cubic k mse", cubic_knearest_mse_avg)
  print("cubic lin mse", cubic_linear_mse_avg)
  print("cubic k/lin mse", cubic_knearest_mse_avg / cubic_linear_mse_avg)

def draw_29_2():
  linear= "1.74226556 1.73714027 1.69178032 1.79227346 1.79891028 1.83092673 1.89674614 1.80451355 1.86817511 1.91332111"\
    .split(" ")
  cubic = "1.54779335 1.64915823 1.61925957 1.65017725 1.65373441 1.73817812 1.68061992 1.76750435 1.78809878 1.8883525" \
    .split(" ")
  linear = [float(v) for v in linear]
  cubic = [float(v) for v in cubic]
  dim_list = range(1, 11)
  plt.plot(dim_list,linear)
  plt.plot(dim_list,cubic)
  print(linear)
  plt.show()

#draw_27()
#linear_img()
#draw_29_2()
#draw_epe_undeter()
draw_29()