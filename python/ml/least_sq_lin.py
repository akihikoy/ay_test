#!/usr/bin/python3
#\file    least_sq_lin.py
#\brief   Least square for linear features.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.26, 2020
from gaussian_rbf import Sq
from least_sq import Rand, GenerateSample
import numpy as np
import numpy.linalg as la
import math
import random


def Func(x):
  return x[0]+2.0*x[1]
  #return x[0]*x[1]
  #return x[0]*math.sin(3.0*x[1])
  #return 3.0-(x[0]*x[0]+x[1]*x[1])
  #return 3.0-(x[0]*x[0]+Sq(math.sin(3.0*x[1])))
  #return 1.0 if x[0]*x[0]+x[1]*x[1] < 0.25 else 0.0

#f_reg: regularization
def GetWeightByLeastSq(data_x, data_f, f_reg=0.1, with_std=False):
  V= np.array(data_f)
  Theta= np.array([[1.0]+x for x in data_x])
  w= la.inv(Theta.T.dot(Theta)+f_reg*np.eye(Theta.shape[1])).dot(Theta.T).dot(V)
  if not with_std:  return w
  std= np.std(V-Theta.dot(w))
  return w,std


#def PrintEq(s):  print '%s= %r' % (s, eval(s))

if __name__=='__main__':
  import time
  xmin= [-1.,-1.]
  xmax= [2.,3.]

  t0= time.time()
  data_x, data_f= GenerateSample(xmin, xmax, N_sample=300, Func=Func, noise=1.0)
  print('GenerateSample/Computation time:',time.time()-t0)
  t0= time.time()
  w,std= GetWeightByLeastSq(data_x, data_f, with_std=True)
  print('GetWeightByLeastSq/Computation time:',time.time()-t0)
  print(w,std)


  fp= open('/tmp/data.dat','w')
  for x,f in zip(data_x, data_f):
    fp.write('%f %f %f\n' % (x[0],x[1], f))

  t0= time.time()
  fp= open('/tmp/approx.dat','w')
  for x0 in np.arange(xmin[0],xmax[0],(xmax[0]-xmin[0])/50.0):
    for x1 in np.arange(xmin[1],xmax[1],(xmax[1]-xmin[1])/50.0):
      x= [x0,x1]
      f= w.T.dot([1.0]+x)
      fp.write('%f %f %f %f %f\n' % (x0,x1, f,f-std,f+std))
    fp.write('\n')
  print('Plotting data/Computation time:',time.time()-t0)

  print('qplot -x -3d /tmp/approx.dat w l /tmp/approx.dat u 1:2:4 w l /tmp/approx.dat u 1:2:5 w l /tmp/data.dat')

