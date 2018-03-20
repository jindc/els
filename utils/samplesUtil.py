import numpy as np
from numpy import *
from numpy import linalg
import math


def radial(sample_x):
  return np.power(np.e, -8 * np.power(linalg.norm(sample_x, ord=2, axis=1), 2))

def one_pow_func(samples_x):
  ret = [ math.pow(math.e, -8 *  math.pow(v[0],2)) for v in samples_x]
  return ret

def cubic_with_dim0(sample_x):
  return [ 0.5 * math.pow(v[0]+1,3) for v in sample_x ]

def linear_undeter(sample_x):
  return [ v[0] + np.random.normal() for v in sample_x  ]

def cubic_with_gauss_dim0(sample_x,mean=0,var=1):
  return [0.5 * math.pow(v[0] + 1, 3) + var * np.random.normal()+mean for v in sample_x]

def generate_sample_with_gauss_dim0(sample_x,mean=0,var=1):
  return [v[0] + var * np.random.normal()+mean for v in sample_x]