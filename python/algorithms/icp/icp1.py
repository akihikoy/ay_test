#!/usr/bin/python
#\file    icp1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.11, 2020
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

def Dist(seq1,seq2, dt,f0,f1):
  seq2_mod= [[t+dt,f0+f1*f] for t,f in seq2]
  nbrs= sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto').fit(seq2_mod)
  distances,indices= nbrs.kneighbors(seq1)
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
  nbrs= sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto').fit(seq2_mod)
  _,indices= nbrs.kneighbors(seq1)
  with open('/tmp/correspond.dat','w') as fp:
    for i1,idx in enumerate(indices):
      i2= idx[0]
      fp.write('%f %f\n' % (seq1[i1][0],seq1[i1][1]))
      fp.write('%f %f\n' % (seq2_mod[i2][0],seq2_mod[i2][1]))
      fp.write('\n')
  print('Plot by:')
  print('  qplot -x /tmp/seq1.dat w l /tmp/seq2.dat u 1:\'($2*50)\' w l /tmp/seq2_mod.dat w l')
  print('  qplot -x /tmp/seq1.dat w l /tmp/seq2_mod.dat w l /tmp/correspond.dat w lp')

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

