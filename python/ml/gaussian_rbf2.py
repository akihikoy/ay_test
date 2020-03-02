#!/usr/bin/python
import math
import numpy as np
import numpy.linalg as la

def f(x):
  return x[0]*math.sin(3.0*x[1])

#Uniform noise in [-1,1]
rand= lambda: 2.0*(-0.5+random.random())

data_x= [[2.0*rand(), 2.0*rand()] for i in range(200)]
data_y= [[f(x)+0.1*rand()] for x in data_x]

def Sq(x):
  return x*x

def Vec(x):
  return np.array(x)

#NOTE: x, mu, invSig should be np.array
def GaussianN(x, mu, invSig):
  ex= math.exp(-0.5*(x-mu).T.dot(invSig).dot(x-mu))
  return ex

#Feature vector with Gaussian basis function
#NOTE: x, mu, invSig should be np.array
def FeaturesG(x, mu, invSig, same_Sig=True):
  if same_Sig:  return [GaussianN(x,mu[d],invSig) for d in range(len(mu))]
  else:         return [GaussianN(x,mu[d],invSig[d]) for d in range(len(mu))]

#NOTE: x, mu, invSig should be np.array
def Quadratic(x, mu, invSig):
  return 0.5*(x-mu).T.dot(invSig).dot(x-mu)

#Feature vector with normalized Gaussian basis function
#NOTE: x, mu, invSig should be np.array
def FeaturesNG(x, mu, invSig, same_Sig=True):
  if same_Sig:  quad= [Quadratic(x,mu[d],invSig) for d in range(len(mu))]
  else:         quad= [Quadratic(x,mu[d],invSig[d]) for d in range(len(mu))]
  quad_max= max(quad)
  gaussian= [math.exp(quad_max-q) for q in quad]
  sum_g= sum(gaussian)  #Should be greater than 1.0
  return [g/sum_g for g in gaussian]


def ConstructRBF(xmin, xmax, n_units):
  dim= len(xmin)
  assert(dim==len(xmax))
  assert(dim==len(n_units))
  #np.mgrid[-1:1:3j,-1:1:3j,-1:1:3j].reshape([3,3**3]).T
  #np.array(np.meshgrid(np.mgrid[-1:1:3j],np.mgrid[-2:2:3j],np.mgrid[-1:1:3j],sparse=False)).reshape(3,3*3*3)
  disc= TDiscretizer(xmin, xmax, [n-1 for n in n_units])
  mu= np.array([[0.0]*dim]*disc.Size())
  for i,v in enumerate(disc.VecSet()):
    mu[i][:]= Vec(v)
  invSig= np.array([[0.0]*dim]*dim)
  for d in range(dim):
    if xmax[d]!=xmin[d]:
      invSig[d,d]= Sq(1.0/(0.75*(xmax[d]-xmin[d])/float(n_units[d]-1)/2.0))
    else:
      invSig[d,d]= 1.0e-6
  return mu, invSig

if __name__=='__main__':
  import random
  xmin= [-1.,-1.]
  xmax= [-2.,-3.]
  n_units= [5,5]
  mu, invSig= ConstructRBF(xmin, xmax, n_units)
  #Features= FeaturesG
  Features= FeaturesNG

  w= Vec([random.random() for d in range(len(mu))])
  #w= Vec([1.0 for d in range(len(mu))])

  for x0 in np.arange(xmin[0],xmax[0],(xmax[0]-xmin[0])/50.0):
    for x1 in np.arange(xmin[1],xmax[1],(xmax[1]-xmin[1])/50.0):
      x= Vec([x0,x1])
      f= w.T.dot(Features(x, mu, invSig))
      print '%f %f %f' % (x0,x1, f)
    print ''

