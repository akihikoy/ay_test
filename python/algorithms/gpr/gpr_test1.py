#!/usr/bin/python

import numpy as np

def Mat(x):
  return np.mat(x)

def Gaussian(x, x0, c, a=1.0):
  diff= x - x0
  dot_product= diff * diff.T
  return a * np.exp(dot_product / (-2.0 * c**2))

class TGPR:
  def __init__(self):
    pass
  #Get kernel k(x)=[k(x,x1),k(x,x2),...]^T
  def Kernel(self,x):
    N= self.X.shape[0]
    k= Mat([0.0]*N).T
    for n in range(N):
      k[n]= Gaussian(x,self.X[n],self.C)
    return k
  #Train with data X,Y.
  #   [x1^T]    [y1^T]
  # X=[x2^T]  Y=[y2^T]
  #   [... ]    [... ]
  # c: Gaussian kernel width.
  # f_reg: Reguralization factor.
  def Train(self,X,Y,c=1.0,f_reg=0.01):
    self.X= Mat(X)
    self.Y= Mat(Y)
    self.C= c
    N= self.X.shape[0]
    self.K= Mat([[0.0]*N]*N)
    for n in range(N):
      self.K[:,n]= self.Kernel(self.X[n])
    self.InvK= (self.K+f_reg*np.eye(N,N)).I
  def Predict(self,x):
    return (self.Kernel(x).T * self.InvK * self.Y).T


import random

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin


if __name__=='__main__':
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))
  import math
  true_func= lambda x: 1.2+math.sin(x)
  data_x= [[x+1.0*Rand()] for x in FRange1(-3.0,5.0,10)]  # ,1.0 is to learn const
  data_y= [[true_func(x[0])+0.3*Rand()] for x in data_x]

  fp1= file('/tmp/smpl.dat','w')
  for x,y in zip(data_x,data_y):
    fp1.write('%f %f\n' % (x[0],y[0]))
  fp1.close()

  gpr= TGPR()
  gpr.Train(data_x, data_y, c=0.5)

  fp1= file('/tmp/true.dat','w')
  fp2= file('/tmp/est.dat','w')
  for x in FRange1(-7.0,10.0,200):
    y= gpr.Predict(np.mat([x]))  # ,1.0 is to learn const
    fp1.write('%f %f\n' % (x,true_func(x)))
    fp2.write('%f %f\n' % (x,y))
  fp1.close()
  fp2.close()

  print 'Plot by:'
  print 'qplot -x /tmp/est.dat w l /tmp/true.dat w l /tmp/smpl.dat w p'

