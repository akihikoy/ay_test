#!/usr/bin/python3
#\file    icp2.py
#\brief   Iterative Closest Point; two-mode linear model (with data scaling)
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.12, 2020
import numpy as np
import sklearn.neighbors
import scipy.optimize

def LoadTFSeq(filename):
  seq= []
  with open(filename,'r') as fp:
    while True:
      line= fp.readline()
      if not line: break
      values= list(map(float,line.split()))
      seq.append([values[0],values[1]])
  return seq

conv_func= lambda: None

Sigmoid= lambda x,xt,beta: 1.0/(1.0+np.exp(-beta*(x-xt)))
LogSigmoid= lambda x,xt,beta: (lambda xx:np.log(1.0/(1.0+np.exp(xx))) if xx<0.0 else -xx+np.log(1.0/(1.0+np.exp(-xx))))(-beta*(x-xt))
#Two-mode linear model (parameter p=[xt,beta,f10,f11,f21])
F2MLin= lambda x,p: (p[3]-p[4])/p[1]*LogSigmoid(x,p[0],p[1])+p[4]*x+(p[2]-(p[4]-p[3])*p[0])
#Two-mode linear model crossing the origin (parameter p=[xt,beta,f11,f21])
F2MLin0= lambda x,p: F2MLin(x,[p[0],p[1],(p[3]-p[2])/p[1]*(LogSigmoid(0.0,p[0],p[1])+p[0]*p[1]),p[2],p[3]])
def SetupModel(model):
  global conv_func
  if model=='linear':  #Linear model (parameter p=[f0,f1])
    conv_func= lambda t,f,dt,p: [t+dt,p[0]+p[1]*f]
  elif model=='2mlin':  #Two-mode linear model (parameter p=[xt,beta,f10,f11,f21])
    conv_func= lambda t,f,dt,p: [t+dt,F2MLin(f,p)]
  elif model=='2mlin0':  #Two-mode linear model crossing origin (parameter p=[xt,beta,f11,f21])
    conv_func= lambda t,f,dt,p: [t+dt,F2MLin0(f,p)]

def KNN(seq1,seq2):
  s= 1.0/np.max(seq1,axis=0)  #Scaling parameters.
  nbrs= sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto', metric='wminkowski', metric_params={'w':s}).fit(seq2)
  return nbrs.kneighbors(seq1)

def Plot(seq1,seq2, dt,p):
  seq2_mod= [conv_func(t,f,dt,p) for t,f in seq2]
  with open('/tmp/seq1.dat','w') as fp:
    for t,f in seq1:
      fp.write('%f %f\n' % (t,f))
  with open('/tmp/seq2.dat','w') as fp:
    for t,f in seq2:
      fp.write('%f %f\n' % (t,f))
  with open('/tmp/seq2_mod.dat','w') as fp:
    for t,f in seq2_mod:
      fp.write('%f %f\n' % (t,f))
  _,indices= KNN(seq1,seq2_mod)
  with open('/tmp/correspond.dat','w') as fp:
    for i1,idx in enumerate(indices):
      i2= idx[0]
      fp.write('%f %f\n' % (seq1[i1][0],seq1[i1][1]))
      fp.write('%f %f\n' % (seq2_mod[i2][0],seq2_mod[i2][1]))
      fp.write('\n')
  seq1_reg= sklearn.neighbors.KNeighborsRegressor(2, weights='distance').fit(np.mat(seq1)[:,0], np.mat(seq1)[:,1])
  with open('/tmp/seq1_seq2_seq2_mod.dat','w') as fp:
    for (t,f),(t_,f_mod) in zip(seq2,seq2_mod):
      fp.write('%f %f %f %f\n' % (t,seq1_reg.predict([[t]])[0][0],f,f_mod))
  with open('/tmp/model.dat','w') as fp:
    for f in np.arange(np.min(seq2,axis=0)[1],np.max(seq2,axis=0)[1],(np.max(seq2,axis=0)[1]-np.min(seq2,axis=0)[1])/50.0):
      fp.write('%f %f\n' % (f,conv_func(0.0,f,dt,p)[1]))
  print('#Plot by:')
  print('qplot -x /tmp/seq1.dat w l /tmp/seq2.dat u 1:\'($2*50)\' w l /tmp/seq2_mod.dat w l')
  print('qplot -x /tmp/seq1.dat w l /tmp/seq2_mod.dat w l /tmp/correspond.dat w lp')
  print('qplot -x /tmp/seq1_seq2_seq2_mod.dat w l /tmp/seq1.dat w p')
  print('qplot -x -s "set size ratio -1" "x" w l /tmp/seq1_seq2_seq2_mod.dat u 4:2 w lp')
  print('qplot -x /tmp/model.dat w l /tmp/seq1_seq2_seq2_mod.dat u 3:2 w lp')

def Dist(seq1,seq2, dt,p):
  seq2_mod= [conv_func(t,f,dt,p) for t,f in seq2]
  #distances,_= KNN(seq1,seq2_mod)
  distances,_= KNN(seq2_mod,seq1)
  print(p,sum(distances))
  return sum(distances)

if __name__=='__main__':
  seq1= LoadTFSeq('../../data/time_f1_001.dat')
  seq2= LoadTFSeq('../../data/time_f2_001.dat')
  #seq1= LoadTFSeq('../../data/time_f1_002.dat')
  #seq2= LoadTFSeq('../../data/time_f2_002.dat')

  #model,pm= 'linear',True
  #model,pm= '2mlin',True
  model,pm= '2mlin0',True
  #xmin= [-0.0, -10.0, 0.0]
  #xmax= [0.5, 10.0, 100.0]
  SetupModel(model)
  grad_max= 1000.0
  if model=='linear':  #Linear model (parameter p=[f0,f1])
    xmin= [-0.0, -100.0, 0.0 if pm else -grad_max]
    xmax= [0.5, 100.0, grad_max if pm else 0.0]
    popsize= 10
  elif model=='2mlin':  #Two-mode linear model (parameter p=[xt,beta,f10,f11,f21])
    xmin= [-0.0, np.min(seq2,axis=0)[1], 100.0, -100.0, 0.0 if pm else -grad_max, 0.0 if pm else -grad_max]
    xmax= [0.5, np.max(seq2,axis=0)[1], 1000.0, 100.0, grad_max if pm else 0.0, grad_max if pm else 0.0]
    popsize= 10
  elif model=='2mlin0':  #Two-mode linear model crossing origin (parameter p=[xt,beta,f11,f21])
    xmin= [-0.0, np.min(seq2,axis=0)[1], 100.0, 0.0 if pm else -grad_max, 0.0 if pm else -grad_max]
    xmax= [0.5, np.max(seq2,axis=0)[1], 1000.0, grad_max if pm else 0.0, grad_max if pm else 0.0]
    popsize= 10

  tol= 1.0e-4
  res= scipy.optimize.differential_evolution(lambda x:Dist(seq1,seq2,x[0],x[1:]), np.array([xmin,xmax]).T, strategy='best1bin', maxiter=30, popsize=popsize, tol=tol, mutation=(0.5, 1), recombination=0.7)
  print(res)
  print(('Result=',res.x,Dist(seq1,seq2,res.x[0],res.x[1:])))
  Plot(seq1,seq2,res.x[0],res.x[1:])

