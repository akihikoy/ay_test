#!/usr/bin/python3
'''
Incremental version of LWR; locally weighted regression.
'''

import numpy as np
import numpy.linalg as la

#Distance of two vectors: L2 norm of their difference
def Dist(p1,p2):
  return la.norm(np.array(p2)-p1)

def Mat(x):
  return np.mat(x)

def Gaussian(x, x0, c, a=1.0):
  diff= x - x0
  dot_product= diff * diff.T
  return a * np.exp(dot_product / (-2.0 * c**2))

class TLWR:
  def __init__(self):
    self.LazyCopy= False

  #Get weights W(x)=diag([k(x,x1),k(x,x2),...])
  def Weights(self,x):
    N= self.X.shape[0]
    W= Mat(np.eye(N,N))
    for n in range(N):
      W[n,n]= Gaussian(x,self.X[n],self.C[n])
    return W

  #Train with data X,Y.
  #   [x1^T]    [y1^T]
  # X=[x2^T]  Y=[y2^T]
  #   [... ]    [... ]
  # c: Gaussian kernel width.
  # f_reg: Reguralization factor.
  def Train(self,X,Y,c_min=0.01,f_reg=0.01):
    self.X= Mat(X)
    self.Y= Mat(Y)
    self.C= self.AutoWidth(c_min)
    self.FReg= f_reg

  #Initialization for incremental learning
  def Init(self,c_min=0.01, c_gain=0.7, f_reg=0.01):
    self.DataX= []
    self.DataY= []
    self.C= []
    self.Closests= []
    self.CMin= c_min
    self.CGain= c_gain
    self.FReg= f_reg
    self.LazyCopy= False

  #Incrementally update the internal parameters
  def Update(self,x,y):
    #FIXME: it's better to use an efficient nearest neighbor like KD tree.
    self.DataX.append(x)
    self.DataY.append(y)
    if len(self.DataX)==1:
      self.Closests.append(-1)
      self.C.append(None)
      return
    n= len(self.Closests)
    self.Closests.append( min([(Dist(x,self.DataX[d]), d) for d in range(n)])[1] )
    self.Closests[self.Closests[-1]]= n
    self.C.append( max(self.CMin, self.CGain*Dist(x,self.DataX[self.Closests[n]])) )
    self.C[self.Closests[-1]]= self.C[-1]
    self.LazyCopy= True

  #Prediction result class.
  class TPredRes:
    def __init__(self):
      self.Y= None  #Prediction result.
      self.Var= None  #Variance matrix.
      self.Grad= None  #Gradient.
  #Do prediction.
  # Return a TPredRes instance.
  # with_var: Whether compute a variance matrix of error at the query point as well.
  # with_grad: Whether compute a gradient at the query point as well.
  def Predict(self,x,with_var=False,with_grad=False):
    if self.LazyCopy:
      self.X= Mat(self.DataX)
      self.Y= Mat(self.DataY)
      self.LazyCopy= False
    res= self.TPredRes()
    D= self.X.shape[1]  #Num of dimensions of x
    W= self.Weights(x)
    beta= (self.X.T*(W*self.X) + self.FReg*np.eye(D,D)).I * (self.X.T*(W*self.Y))
    res.Y= x * beta
    if with_var:
      N= self.X.shape[0]  #Num of samples
      div= W.trace()
      div*= (1.0 - float(D)/float(N)) if N>D else 1.0e-4
      Err= self.X * beta - self.Y
      res.Var= (Err.T * W * Err) / div
    if with_grad:
      res.Grad= beta
    return res

  #Compute Gaussian kernel width for each data point automatically.
  def AutoWidth(self, c_min=0.01, c_max=1.0e6, c_gain=0.7):
    N= self.X.shape[0]
    C= [0.0]*N
    for n in range(N):
      C[n]= max( c_min, min([c_gain*la.norm(self.X[n]-self.X[d]) if d!=n else c_max for d in range(N)]) )
    #print C
    return C

import random

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

def AddOne(x):
  if isinstance(x,list):
    if len(x)==0:  return [1.0]
    if isinstance(x[0],list): return [x[0]+[1.0]]
    return x+[1.0]
  if isinstance(x,np.ndarray):
    return x.tolist()+[1.0]
  if x.shape[0]==1:  return Mat(x.tolist()[0]+[1.0])
  if x.shape[1]==1:  return Mat(x.T.tolist()[0]+[1.0]).T
  return None

def AddOnes(X):
  if isinstance(X,list):  Y= X
  else:  Y= X.tolist()
  for x in Y:
    x.append(1.0)
  return Y


if __name__=='__main__':
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))
  import math

  example= 1
  #example= 2

  if example==1:
    true_func= lambda x: 1.2+math.sin(x)
    data_x= [[x+1.0*Rand()] for x in FRange1(-3.0,5.0,10)]
    data_y= [[true_func(x[0])+0.3*Rand()] for x in data_x]

    fp1= open('/tmp/smpl.dat','w')
    for x,y in zip(data_x,data_y):
      fp1.write('%f %f\n' % (x[0],y[0]))
    fp1.close()

    lwr= TLWR()
    #lwr.Train(AddOnes(data_x), data_y, c_min=0.3)
    lwr.Init(c_min=0.3)
    for x,y in zip(data_x,data_y):
      lwr.Update(AddOne(x),y)

    fp1= open('/tmp/true.dat','w')
    fp2= open('/tmp/est.dat','w')
    for x in FRange1(-7.0,10.0,200):
      pred= lwr.Predict(AddOne([x]),with_var=True,with_grad=True)
      #print 'pred.Grad=',pred.Grad
      fp1.write('%f %f\n' % (x,true_func(x)))
      fp2.write('%f %f %f %f %f\n' % (x,pred.Y,np.sqrt(pred.Var),1.0,pred.Grad[0]))
    fp1.close()
    fp2.close()

    print('Plot by:')
    print('qplot -x /tmp/est.dat w errorbars /tmp/true.dat w l /tmp/smpl.dat w p')

  elif example==2:
    true_func= lambda x: 1.2+math.sin(2.0*x[0])*x[1]
    data_x= [[4.0*Rand(),4.0*Rand()] for i in range(20)]
    data_y= [[true_func(x)+0.3*Rand()] for x in data_x]

    fp1= open('/tmp/smpl.dat','w')
    for x,y in zip(data_x,data_y):
      fp1.write('%f %f %f\n' % (x[0],x[1],y[0]))
    fp1.close()

    lwr= TLWR()
    #lwr.Train(AddOnes(data_x), data_y, c_min=0.3)
    lwr.Init(c_min=0.3)
    for x,y in zip(data_x,data_y):
      lwr.Update(AddOne(x),y)

    fp1= open('/tmp/true.dat','w')
    fp2= open('/tmp/est.dat','w')
    for x1 in FRange1(-4.0,4.0,50):
      for x2 in FRange1(-4.0,4.0,50):
        y= lwr.Predict(AddOne([x1,x2])).Y
        fp1.write('%f %f %f\n' % (x1,x2,true_func([x1,x2])))
        fp2.write('%f %f %f\n' % (x1,x2,y))
      fp1.write('\n')
      fp2.write('\n')
    fp1.close()
    fp2.close()

    print('Plot by:')
    print('qplot -x -3d /tmp/est.dat w l /tmp/true.dat w l /tmp/smpl.dat w p')
