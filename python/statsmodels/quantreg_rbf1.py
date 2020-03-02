#!/usr/bin/python
#\file    quantreg_rbf1.py
#\brief   Quantile regression with statsmodels
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.29, 2020
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from ml.gaussian_rbf import FeaturesG, FeaturesNG, ConstructRBF


def Func(x):
  #return x[0]
  #return -2.0*x[0]*x[0]+0.5*x[0]-1.0
  return x[0]*np.sin(-3.0*x[0])
  #return 1.0 if x[0]*x[0] < 0.25 else 0.0

def NoiseFunc(x):
  #return np.random.normal(scale=0.01)
  #return np.random.normal(scale=0.2)
  return np.random.normal(scale=0.2*abs(x[0]))

def GenerateSample(xmin, xmax, N_sample, Func, NoiseFunc):
  data_x= [[np.random.uniform(xmin[0],xmax[0])] for i in range(N_sample)]
  data_f= [Func(x)+NoiseFunc(x) for x in data_x]
  return data_x, data_f


if __name__=='__main__':
  import time
  xmin= [-1.]
  xmax= [2.]
  mu,invSig= ConstructRBF(xmin, xmax, [10])

  t0= time.time()
  data_x, data_f= GenerateSample(xmin, xmax, N_sample=300, Func=Func, NoiseFunc=NoiseFunc)
  print 'GenerateSample/Computation time:',time.time()-t0

  t0= time.time()
  Theta= np.array([FeaturesNG(x,mu,invSig) for x in data_x])
  quant_reg= QuantReg(data_f, Theta)
  fit1= quant_reg.fit(q=0.1)
  fit5= quant_reg.fit(q=0.5)
  fit9= quant_reg.fit(q=0.95)
  w1= fit1.params
  w5= fit5.params
  w9= fit9.params
  print fit9.summary()
  print 'Parameters w1:',w1
  print 'Parameters w5:',w5
  print 'Parameters w9:',w9
  print 'QuantReg/Computation time:',time.time()-t0

  with open('/tmp/data.dat','w') as fp:
    for x,f in zip(data_x, data_f):
      fp.write('%f %f\n' % (x[0], f))

  with open('/tmp/approx.dat','w') as fp:
    for x0 in np.arange(xmin[0],xmax[0],(xmax[0]-xmin[0])/200.0):
      x= [x0]
      theta= FeaturesNG(x,mu,invSig)
      f1= w1.T.dot(theta)
      f5= w5.T.dot(theta)
      f9= w9.T.dot(theta)
      fp.write('%f %f %f %f\n' % (x0, f1,f5,f9))

  print 'qplot -x /tmp/approx.dat w l /tmp/approx.dat u 1:3 w l /tmp/approx.dat u 1:4 w l /tmp/data.dat'

