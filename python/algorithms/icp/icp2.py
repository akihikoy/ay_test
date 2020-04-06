#!/usr/bin/python
#\file    icp2.py
#\brief   Iterative Closest Point (with data scaling)
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

def KNN(seq1,seq2):
  s= 1.0/np.max(seq1,axis=0)  #Scaling parameters.
  #d= lambda a,b: np.dot((a-b)**2,s)
  #nbrs= sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto', metric=d).fit(seq2)
  nbrs= sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto', metric='wminkowski', metric_params={'w':s}).fit(seq2)
  return nbrs.kneighbors(seq1)

def Dist(seq1,seq2, dt,f0,f1):
  seq2_mod= [[t+dt,f0+f1*f] for t,f in seq2]
  distances,_= KNN(seq1,seq2_mod)
  print dt,f0,f1, sum(distances)
  return sum(distances)

def Plot(seq1,seq2, dt,f0,f1):
  seq2_mod= [[t+dt,f0+f1*f] for t,f in seq2]
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
      fp.write('%f %f\n' % (f,f0+f1*f))
  print('Plot by:')
  print('  qplot -x /tmp/seq1.dat w l /tmp/seq2.dat u 1:\'($2*50)\' w l /tmp/seq2_mod.dat w l')
  print('  qplot -x /tmp/seq1.dat w l /tmp/seq2_mod.dat w l /tmp/correspond.dat w lp')
  print('  qplot -x /tmp/seq1_seq2_seq2_mod.dat w l /tmp/seq1.dat w p')
  print('  qplot -x -s "set size ratio -1" "x" w l /tmp/seq1_seq2_seq2_mod.dat u 4:2 w lp')
  print('  qplot -x /tmp/model.dat w l /tmp/seq1_seq2_seq2_mod.dat u 3:2 w lp')

if __name__=='__main__':
  #seq1= LoadTFSeq('../../data/time_f1_001.dat')
  #seq2= LoadTFSeq('../../data/time_f2_001.dat')
  seq1= LoadTFSeq('../../data/time_f1_002.dat')
  seq2= LoadTFSeq('../../data/time_f2_002.dat')

  xmin= [-0.0, -10.0, 0.0]
  xmax= [0.5, 10.0, 100.0]
  tol= 1.0e-4
  res= scipy.optimize.differential_evolution(lambda x:Dist(seq1,seq2,x[0],x[1],x[2]), np.array([xmin,xmax]).T, strategy='best1bin', maxiter=300, popsize=10, tol=tol, mutation=(0.5, 1), recombination=0.7)
  print(res)
  print('Result=',res.x,Dist(seq1,seq2,res.x[0],res.x[1],res.x[2]))
  Plot(seq1,seq2,res.x[0],res.x[1],res.x[2])

