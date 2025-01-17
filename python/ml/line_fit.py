#!/usr/bin/python3
import numpy as np
import numpy.linalg as la
import math
import random
import time

def Func(x):
  #return x*math.sin(3.0*x)
  #return 3.0-x*x
  #return 3.0-(x*x+math.sin(3.0*x))
  #return 1.0 if x*x < 0.25 else 0.0
  return 2.9*x + 2.1
  #return -2.9*x - 2.1
  #return 3.1
  #return 4.0*x

def Rand(xmin, xmax):
  return (xmax-xmin)*random.random()+xmin

def GenerateSample(xmin, xmax, N_sample):
  data_x= [Rand(xmin,xmax) for i in range(N_sample)]
  data_f= [Func(x)+0.5*Rand(-1.0,1.0) for x in data_x]
  return data_x, data_f

def LineFitByLeastSq(data_x, data_f, f_reg=0.0):
  V= np.array(data_f)
  Theta= np.array([data_x,[1.0]*len(data_x)]).T
  w= la.inv(Theta.T.dot(Theta)+f_reg*np.eye(Theta.shape[1])).dot(Theta.T).dot(V)
  return w


#def PrintEq(s):  print '%s= %r' % (s, eval(s))

if __name__=='__main__':
  xmin= -1.0
  xmax= 2.0
  data_x, data_f= GenerateSample(xmin, xmax, N_sample=300)
  t0= time.time()
  w= LineFitByLeastSq(data_x, data_f)
  print('Computation time[ms]:',(time.time()-t0)*1000.)
  print('w=',w)

  fp= open('/tmp/data.dat','w')
  for x,f in zip(data_x, data_f):
    fp.write('%f %f\n' % (x, f))

  fp= open('/tmp/approx.dat','w')
  for x in np.arange(xmin,xmax,(xmax-xmin)/50.0):
    f= w.T.dot([x,1.0])
    fp.write('%f %f\n' % (x, f))

  print('qplot -x /tmp/approx.dat w l /tmp/data.dat')

