#!/usr/bin/python3
#\file    dtw_min2.py
#\brief   Minimize DTW to match two sequences.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.11, 2020
# NOTE: Need to install fastdtw by:
#   $ sudo apt-get -f install python3-scipy
#   $ pip install fastdtw
# https://pypi.org/project/fastdtw/
# NOTE: Need to use Python 3 (#!/usr/bin/python3).

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import scipy.optimize

from dtw2 import LoadXFSeq

x_f_seq1= LoadXFSeq('../../data/time_f_z001.dat')
x_f_seq2= LoadXFSeq('../../data/time_f_z002.dat')
#x_f_seq2= LoadXFSeq('../../data/time_f_z003.dat')

def dist(dx,path):
  seq_mod= [[x+dx,f] for x,f in x_f_seq2]
  #d2= sum((euclidean(x_f_seq1[i1],seq_mod[i2]) for i1,i2 in path))
  #d2= sum((abs(x_f_seq1[i1][0]-seq_mod[i2][0]) for i1,i2 in path))
  #d2= sum((abs(x_f_seq1[i1][0]-seq_mod[i2][0])**2 for i1,i2 in path))
  #d2= sum(((x_f_seq1[i1][0]-seq_mod[i2][0])**2+(x_f_seq1[i1][1]-seq_mod[i2][1])**2 for i1,i2 in path))
  d2= sum((np.sqrt((x_f_seq1[i1][0]-seq_mod[i2][0])**2+(x_f_seq1[i1][1]-seq_mod[i2][1])**2) for i1,i2 in path))
  #d2= max(((x_f_seq1[i1][0]-seq_mod[i2][0])**2+(x_f_seq1[i1][1]-seq_mod[i2][1])**2 for i1,i2 in path))
  print((dx,d2))
  return d2

def plot(dx):
  seq_mod= [[x+dx,f] for x,f in x_f_seq2]
  with open('/tmp/seq1.dat','w') as fp:
    for x in x_f_seq1:
      fp.write('%f %f\n' % (x[0],x[1]))
  with open('/tmp/seq2.dat','w') as fp:
    for x in x_f_seq2:
      fp.write('%f %f\n' % (x[0],x[1]))
  with open('/tmp/seq2_mod.dat','w') as fp:
    for x in seq_mod:
      fp.write('%f %f\n' % (x[0],x[1]))
  print('Plot by:')
  print('  qplot -x /tmp/seq1.dat w lp /tmp/seq2.dat w lp /tmp/seq2_mod.dat w lp')

xmin= [-0.1]
xmax= [0.1]
tol= 1.0e-4
#We fix path (the mapping of corresponding points).
_,path= fastdtw(x_f_seq1, x_f_seq2, dist=euclidean)
res= scipy.optimize.differential_evolution(lambda x:dist(x[0],path), np.array([xmin,xmax]).T, strategy='best1bin', maxiter=300, popsize=10, tol=tol, mutation=(0.5, 1), recombination=0.7)
print(res)
print(('Result=',res.x,dist(res.x,path)))
plot(res.x)

