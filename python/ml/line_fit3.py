#!/usr/bin/python
#\file    line_fit3.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.11, 2022
import numpy as np
import numpy.linalg as la
import math
import random
import time
import matplotlib.pyplot as plt

def Func(x):
  return np.vstack((2.9*x + 2.1, -2.9*x - 2.1, 4.0*x, 3.1*np.ones_like(x)))

def Rand(xmin, xmax):
  return (xmax-xmin)*random.random()+xmin

def GenerateSample(xmin, xmax, N_sample):
  data_x= np.random.uniform(xmin,xmax,N_sample)
  data_f= Func(data_x) + np.random.uniform(-0.5,0.5,(4,N_sample))
  return data_x, data_f.T

def LineFit(data_x, data_f):
  w= np.polyfit(data_x, data_f, 1)
  return w


#def PrintEq(s):  print '%s= %r' % (s, eval(s))

if __name__=='__main__':
  xmin= -1.0
  xmax= 2.0
  data_x, data_f= GenerateSample(xmin, xmax, N_sample=300)
  t0= time.time()
  w= LineFit(data_x, data_f)
  print 'Computation time[ms]:',(time.time()-t0)*1000.
  print 'w=',w

  test_x= np.arange(xmin,xmax,(xmax-xmin)/50.0)
  test_f= np.dot(np.vstack((test_x,np.ones_like(test_x))).T,w)

  markers= ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
  fig= plt.figure(figsize=(5,5))
  ax= fig.add_subplot(1,1,1,title='line_fit3',xlabel='x',ylabel='y')
  for i in range(4):
    ax.scatter(data_x, data_f[:,i], color='blue', marker=markers[i], s=10, label='data-{}'.format(i))
    ax.plot(test_x, test_f[:,i], color='red', linewidth=2, label='fit-{}'.format(i))
  ax.legend()
  plt.show()
