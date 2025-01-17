#!/usr/bin/python3
#Online linear regression using Gaussian RBF.

from gaussian_rbf import Sq, Vec, FeaturesG, FeaturesNG, ConstructRBF
import numpy as np
import numpy.linalg as la
import math
import random

#Features= FeaturesG
Features= FeaturesNG

def Func(x):
  return x[0]*math.sin(3.0*x[1])
  #return 3.0-(x[0]*x[0]+x[1]*x[1])
  #return 3.0-(x[0]*x[0]+Sq(math.sin(3.0*x[1])))
  #return 1.0 if x[0]*x[0]+x[1]*x[1] < 0.25 else 0.0

def Rand(xmin, xmax):
  return (xmax-xmin)*random.random()+xmin

def UpdateWeight(w, x, f, func_feat, alpha=0.5):
  feat= Vec(func_feat(x))
  err= f - w.T.dot(feat)
  w+= alpha*err*feat
  #print la.norm(1.0/(feat+0.1))
  #w+= alpha*err/(feat+0.1)

#def PrintEq(s):  print '%s= %r' % (s, eval(s))

if __name__=='__main__':
  xmin= [-1.,-1.]
  xmax= [2.,3.]
  n_units= [6]*2
  mu, invSig= ConstructRBF(xmin, xmax, n_units)

  data_x= []
  data_f= []
  N_sample= 200
  w= Vec([0.0]*len(mu))
  for i in range(N_sample):
    x= [Rand(xmin[0],xmax[0]), Rand(xmin[1],xmax[1])]
    f= Func(x)
    UpdateWeight(w, x, f, func_feat=lambda x: Features(x,mu,invSig))
    data_x.append(x)
    data_f.append(f)

  fp= open('res/data.dat','w')
  for x,f in zip(data_x, data_f):
    fp.write('%f %f %f\n' % (x[0],x[1], f))

  fp= open('res/approx.dat','w')
  for x0 in np.arange(xmin[0],xmax[0],(xmax[0]-xmin[0])/50.0):
    for x1 in np.arange(xmin[1],xmax[1],(xmax[1]-xmin[1])/50.0):
      x= Vec([x0,x1])
      f= w.T.dot(Features(x, mu, invSig))
      fp.write('%f %f %f\n' % (x0,x1, f))
    fp.write('\n')

  print('qplot -x -3d res/approx.dat w l res/data.dat')

