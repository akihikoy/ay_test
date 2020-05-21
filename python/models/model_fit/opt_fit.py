#!/usr/bin/python
#\file    icp_fit1.py
#\brief   Fitting function with an optimizer (two-mode linear model).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.18, 2020
import numpy as np
import sklearn.neighbors
import scipy.optimize

model_func= lambda: None
Sigmoid= lambda x,xt,beta: 1.0/(1.0+np.exp(-beta*(x-xt)))
LogSigmoid= lambda x,xt,beta: (lambda xx:np.log(1.0/(1.0+np.exp(xx))) if xx<0.0 else -xx+np.log(1.0/(1.0+np.exp(-xx))))(-beta*(x-xt))
#Two-mode linear model (parameter p=[xt,beta,f10,f11,f21])
F2MLin= lambda x,p: (p[3]-p[4])/p[1]*LogSigmoid(x,p[0],p[1])+p[4]*x+(p[2]-(p[4]-p[3])*p[0])
#Two-mode linear model crossing the origin (parameter p=[xt,beta,f11,f21])
F2MLin0= lambda x,p: F2MLin(x,[p[0],p[1],(p[3]-p[2])/p[1]*(LogSigmoid(0.0,p[0],p[1])+p[0]*p[1]),p[2],p[3]])
def SetupModel(model):
  global model_func
  if model=='linear':  #Linear model (parameter p=[f0,f1])
    model_func= lambda x,p: p[0]+p[1]*x
  elif model=='2mlin':  #Two-mode linear model (parameter p=[xt,beta,f10,f11,f21])
    model_func= lambda x,p: F2MLin(x,p)
  elif model=='2mlin0':  #Two-mode linear model crossing origin (parameter p=[xt,beta,f11,f21])
    model_func= lambda x,p: F2MLin0(x,p)

def Error(seq,p):
  return sum((y-model_func(x,p))**2 for x,y in seq)

def Plot(seq, p):
  with open('/tmp/seq.dat','w') as fp:
    for x,y in seq:
      fp.write('%f %f\n' % (x,y))
  with open('/tmp/model.dat','w') as fp:
    for x in np.arange(np.min(seq,axis=0)[0],np.max(seq,axis=0)[0],(np.max(seq,axis=0)[0]-np.min(seq,axis=0)[0])/200.0):
      fp.write('%f %f\n' % (x,model_func(x,p)))
  print('#Plot by:')
  print('qplot -x /tmp/seq.dat w p /tmp/model.dat w l')

def LoadSeq(filename,e1=0,e2=1):
  seq= []
  with open(filename,'r') as fp:
    while True:
      line= fp.readline()
      if not line: break
      values= list(map(float,line.split()))
      seq.append([values[e1],values[e2]])
  return seq

if __name__=='__main__':
  #seq= LoadSeq('../../data/time_f1_001.dat')
  #seq= LoadSeq('../../data/time_f1_002.dat')
  seq= LoadSeq('../../data/time_f_z003.dat',2,1)

  #model= 'linear'
  model= '2mlin'
  #model= '2mlin0'
  SetupModel(model)
  bias_max=1.0e7
  grad_max= 1.0e5
  if model=='linear':  #Linear model (parameter p=[f0,f1])
    xmin= [-bias_max, -grad_max]
    xmax= [ bias_max,  grad_max]
    popsize= 10
  elif model=='2mlin':  #Two-mode linear model (parameter p=[xt,beta,f10,f11,f21])
    xmin= [np.min(seq,axis=0)[0], 0.001, -bias_max, -grad_max, -grad_max]
    xmax= [np.max(seq,axis=0)[0], 1.0e7,  bias_max,  grad_max,  grad_max]
    popsize= 10
  elif model=='2mlin0':  #Two-mode linear model crossing origin (parameter p=[xt,beta,f11,f21])
    xmin= [np.min(seq,axis=0)[0], 0.001, -grad_max, -grad_max]
    xmax= [np.max(seq,axis=0)[0], 1.0e7,  grad_max,  grad_max]
    popsize= 10

  tol= 1.0e-4
  res= scipy.optimize.differential_evolution(lambda x:Error(seq,x), np.array([xmin,xmax]).T, strategy='best1bin', maxiter=300, popsize=popsize, tol=tol, mutation=(0.5, 1), recombination=0.7)
  print(res)
  print('Result=',res.x,Error(seq,res.x))
  Plot(seq,res.x)

