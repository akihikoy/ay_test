#!/usr/bin/python3

import numpy as np
import numpy.linalg as la

from gpr_test2 import *

def SerializeMat(m):
  return m.reshape(1,m.shape[0]*m.shape[1]).tolist()[0]

#Get gradient Dy/Dx from sample Y=[y_1,y_2,...], X=[x_1,x_2,...] where x_t and y_t correspond.
#Another idea is gradually updating the gradient.
def GradientFromSample(Y,X, maxg=100.0, dx_ratio=0.3):  # ,mindx=1.0e-6
  assert(len(Y)==len(X))
  assert(len(X)>=2)
  G= Mat([[0.0]*len(Y[0])]*len(X[0]))  #rows= len(X[0]), cols= len(Y[0])
  N= Mat([[0]*len(Y[0])]*len(X[0]))
  for t in range(len(X)-1):
    Dx= Mat([0.0]*len(X[0]))
    Dy= Mat([0.0]*len(Y[0]))
    for ix in range(len(X[0])):  Dx[ix]= X[t+1][ix]-X[t][ix]
    for iy in range(len(Y[0])):  Dy[iy]= Y[t+1][iy]-Y[t][iy]
    mindx= dx_ratio * la.norm(Dx)
    for ix in range(len(X[0])):
      for iy in range(len(Y[0])):
        dy= Dy[iy]
        dx= Dx[ix]
        #FIXME:Why not la.norm(g[:,iy])<maxg ???
        if abs(dx)>mindx:
          dydx= dy/dx
          if abs(dydx)<maxg:
            G[ix,iy]+= dydx
            N[ix,iy]+= 1
          else:
            G[ix,iy]+= dydx*maxg/abs(dydx)
            N[ix,iy]+= 1

  for ix in range(len(X[0])):
    for iy in range(len(Y[0])):
      if N[ix,iy]>0:  G[ix,iy]/= float(N[ix,iy])
  if len(Y[0])==1:  return G[:,0]
  if len(X[0])==1:  return G[0,:]
  return G

def GetLinearModels(data_x, data_y):
  X= Mat(data_x)
  Y= Mat(data_y)
  N= X.shape[0]

  closests= [0.0]*N
  for n in range(N):
    closests[n]= min([(la.norm(X[n]-X[d]) if d!=n else 1.0e+10, d) for d in range(N)])[1]

  data_lin= []
  for n in range(N):
    g= GradientFromSample([Y[n],Y[closests[n]]], [X[n],X[closests[n]]])
    b= Y[n]-g*X[n]
    data_lin.append(SerializeMat(g)+SerializeMat(b))  #WARNING:Only works for dim(y)==1
  return data_lin

class TGPRLin:
  def __init__(self):
    self.gpr= TGPR2()
  #Train with data X,Y.
  #   [x1^T]    [y1^T]
  # X=[x2^T]  Y=[y2^T]
  #   [... ]    [... ]
  # c: Gaussian kernel width.
  # f_reg: Reguralization factor.
  def Train(self,X,Y,c_min=0.01,f_reg=0.01):
    self.X= Mat(X)
    self.Y= Mat(Y)
    self.L= Mat(GetLinearModels(self.X, self.Y))
    #print self.L
    self.gpr.Train(self.X,self.L,c_min,f_reg)
  def Predict(self,x):
    l= self.gpr.Predict(x)
    return Mat(l[:-1])*Mat(x) + Mat(l[-1])  #WARNING:Only works for dim(y)==1


if __name__=='__main__':
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))
  import math
  true_func= lambda x: 1.2+math.sin(x)
  data_x= [[x+1.0*Rand()] for x in FRange1(-3.0,5.0,2)]
  data_y= [[true_func(x[0])+0.3*Rand()] for x in data_x]

  fp1= open('/tmp/smpl.dat','w')
  for x,y in zip(data_x,data_y):
    fp1.write('%f %f\n' % (x[0],y[0]))
  fp1.close()

  gpr= TGPRLin()
  gpr.Train(data_x, data_y, c_min=0.5)

  fp1= open('/tmp/true.dat','w')
  fp2= open('/tmp/est.dat','w')
  for x in FRange1(-7.0,10.0,200):
    y= gpr.Predict(np.mat([x]))
    fp1.write('%f %f\n' % (x,true_func(x)))
    fp2.write('%f %f\n' % (x,y))
  fp1.close()
  fp2.close()

  print('Plot by:')
  print('qplot -x /tmp/est.dat w l /tmp/true.dat w l /tmp/smpl.dat w p')

