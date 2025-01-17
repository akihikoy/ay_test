#!/usr/bin/python3
from gaussian_rbf import Sq, Vec, FeaturesG, FeaturesNG, ConstructRBF
from least_sq import Rand, GetWeightByLeastSq
import numpy as np
import numpy.linalg as la
import math
import random

#Features= FeaturesG
Features= FeaturesNG

def Func(x):
  #return x[0]
  #return -2.0*x[0]*x[0]+0.5*x[0]-1.0
  return x[0]*np.sin(-3.0*x[0])
  #return 1.0 if x[0]*x[0] < 0.25 else 0.0

def Rand(xmin, xmax):
  return (xmax-xmin)*random.random()+xmin

def GenerateSample(xmin, xmax, N_sample, Func, noise=1.0e-10):
  data_x= [[Rand(xmin[0],xmax[0])] for i in range(N_sample)]
  data_f= [Func(x)+np.random.normal(scale=noise) for x in data_x]
  return data_x, data_f

#def PrintEq(s):  print '%s= %r' % (s, eval(s))

if __name__=='__main__':
  import time
  xmin= [-1.]
  xmax= [2.]
  t0= time.time()
  mu,invSig= ConstructRBF(xmin, xmax, [10])
  print('ConstructRBF/Computation time:',time.time()-t0)

  t0= time.time()
  data_x, data_f= GenerateSample(xmin, xmax, N_sample=300, Func=Func)
  print('GenerateSample/Computation time:',time.time()-t0)
  t0= time.time()
  w= GetWeightByLeastSq(data_x, data_f, func_feat=lambda x: Features(x,mu,invSig))
  print('GetWeightByLeastSq/Computation time:',time.time()-t0)


  with open('/tmp/data.dat','w') as fp:
    for x,f in zip(data_x, data_f):
      fp.write('%f %f\n' % (x[0], f))

  t0= time.time()
  with open ('/tmp/approx.dat','w') as fp:
    for x0 in np.arange(xmin[0],xmax[0],(xmax[0]-xmin[0])/200.0):
      x= Vec([x0])
      f= w.T.dot(Features(x, mu, invSig))
      fp.write('%f %f\n' % (x0, f))
  print('Plotting data/Computation time:',time.time()-t0)

  print('qplot -x /tmp/approx.dat w l /tmp/data.dat')

