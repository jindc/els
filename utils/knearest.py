import numpy as np
from numpy import *
from numpy import linalg
from matplotlib import pyplot as plt
from IPython.display import display
import math


class Knearest:
  X = None
  Y = None
  beta = None
  beta = None
  def fit(self,X,Y):
    self.X = np.array(X)
    self.Y = np.array(Y)
    return self

  def predict(self,features,k=1):
    ret = []
    for sample in features.tolist():
      dlist = [(self.X[i], self.Y[i], np.linalg.norm(np.array([self.X[i]]) - np.array([sample]),axis=1).tolist()[0]) for i in
             range(len(self.X))]
      dlist.sort(key=lambda item: item[2])
      avg = math.fsum([ v[1] for v in dlist[:k]  ])/np.min([k,len(dlist)])
      ret.append(avg)
    # print( dlist)
    return ret
