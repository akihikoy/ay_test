#!/usr/bin/python3
#\file    knn_match2.py
#\brief   Nearest neighbor test (with scaling).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.12, 2020
import numpy as np
import sklearn.neighbors

def Func(x):
  return 1.0+x*(-3.2+x*(-0.3+1.5*x))
  #return 0.3*(1.0+x*(-3.2+x*(-0.3+1.5*x)))

def KNNMatch1(seq1,seq2,k=1,filename=None):
  s= 1.0/np.max(seq1,axis=0)  #Scaling parameters.
  #nbrs= sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm='auto').fit(seq2)
  #d= lambda a,b: np.dot((a-b)**2,s)
  #nbrs= sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm='auto', metric=d).fit(seq2)
  nbrs= sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm='auto', metric='wminkowski', metric_params={'w':s}).fit(seq2)
  distances,indices= nbrs.kneighbors(seq1)
  print(indices)
  if filename is not None:
    with open(filename,'w') as fp:
      for i1,i2s in enumerate(indices):
        for i2 in i2s:
          fp.write('{0} {1}\n'.format(seq1[i1][0],seq1[i1][1]))
          fp.write('{0} {1}\n'.format(seq2[i2][0],seq2[i2][1]))
          fp.write('\n')
  return sum(distances)

if __name__=='__main__':
  seq1= [[x,Func(x)] for x in np.arange(-2.0,2.0,0.4)]
  seq2= [(lambda x:[x,Func(x)+np.random.normal(scale=3.0)])(np.random.uniform(-2.0,2.0)) for _ in range(5)]

  with open('/tmp/seq1.dat','w') as fp:
    for x,y in seq1:
      fp.write('{0} {1}\n'.format(x,y))
  with open('/tmp/seq2.dat','w') as fp:
    for x,y in seq2:
      fp.write('{0} {1}\n'.format(x,y))
  print('Plot by:')
  print('$ qplot -x /tmp/seq1.dat w lp /tmp/seq2.dat w p')

  print('KNNMatch1(seq1,seq2,k=1)')
  KNNMatch1(seq1,seq2,k=1,filename='/tmp/knn1.dat')
  print('Plot by:')
  print('$ qplot -x -s \'set size square\' /tmp/seq1.dat w l /tmp/knn1.dat w lp /tmp/seq2.dat w p')

  print('KNNMatch1(seq2,seq1,k=1)')
  KNNMatch1(seq2,seq1,k=1,filename='/tmp/knn2.dat')
  print('Plot by:')
  print('$ qplot -x -s \'set size square\' /tmp/seq1.dat w l /tmp/knn2.dat w lp /tmp/seq2.dat w p')
