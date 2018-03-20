import numpy as np
from numpy import *
from numpy import linalg
from matplotlib import pyplot as plt
from IPython.display import display
import math


class Linear:
  X = None
  Y = None
  beta = None
  def fit(self,X,Y,debug=False):
    self.X = np.array(X)
    self.Y = np.array(Y)

    samples_x_mat = np.mat(np.hstack((np.array(X), np.ones([np.array(X).shape[0], 1]))))
    samples_y_mat = np.mat(Y).reshape([-1, 1])
    self.beta = (samples_x_mat.T * samples_x_mat).I * samples_x_mat.T * samples_y_mat
    if debug:
      print(np.array(X).shape,self.beta)
    return self
  def predict(self,x):
    point_x = np.column_stack((np.array(x), np.ones([np.array(x).shape[0],1])))
    ret =np.mat(point_x) * self.beta
    return ret

