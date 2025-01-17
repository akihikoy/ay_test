#!/usr/bin/python3

import numpy as np

def Mat(x):
  return np.mat(x)

def Gaussian(x, x0, c, a=1.0):
  diff= x - x0
  dot_product= diff * diff.T
  return a * np.exp(dot_product / (-2.0 * c**2))

class TLWR:
  def __init__(self):
    pass
  #Get weights W(x)=diag([k(x,x1),k(x,x2),...])
  def Weights(self,x):
    N= self.X.shape[0]
    W= Mat(np.eye(N,N))
    for n in range(N):
      W[n,n]= Gaussian(x,self.X[n],self.C)
    return W
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
    self.FReg= f_reg
  def Predict(self,x):
    D= self.X.shape[1]
    W= self.Weights(x)
    beta= (self.X.T*(W*self.X) + self.FReg*np.eye(D,D)).I * (self.X.T*(W*self.Y))
    return x * beta

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
  data_x= [[x+1.0*Rand(),1.0] for x in FRange1(-3.0,5.0,10)]  # ,1.0 is to learn const
  data_y= [[true_func(x[0])+0.3*Rand()] for x in data_x]

  fp1= open('/tmp/smpl.dat','w')
  for x,y in zip(data_x,data_y):
    fp1.write('%f %f\n' % (x[0],y[0]))
  fp1.close()

  lwr= TLWR()
  lwr.Train(data_x, data_y, c=0.5)

  fp1= open('/tmp/true.dat','w')
  fp2= open('/tmp/est.dat','w')
  for x in FRange1(-7.0,10.0,200):
    y= lwr.Predict(np.mat([x,1.0]))  # ,1.0 is to learn const
    fp1.write('%f %f\n' % (x,true_func(x)))
    fp2.write('%f %f\n' % (x,y))
  fp1.close()
  fp2.close()

  print('Plot by:')
  print('qplot -x /tmp/est.dat w l /tmp/true.dat w l /tmp/smpl.dat w p')

