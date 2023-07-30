#!/usr/bin/python
#\file    kdtree.py
#\brief   Test of scipy.spatial.KDTree
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.30, 2023

'''
NOTE: KDTree of Python does not implement adding and removing data.
https://stackoverflow.com/questions/17817889/is-there-any-way-to-add-points-to-kd-tree-implementation-in-scipy
https://github.com/scipy/scipy/issues/9029
'''

import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

if __name__=='__main__':
  fig= plt.figure()
  ax1= fig.add_subplot(1,1,1)

  data= np.random.uniform(size=(100,2))*100.0
  print data.shape
  kdtree= KDTree(data, leafsize=10)
  p_q= np.random.uniform(size=2)*100.0
  d,i= kdtree.query(p_q)
  p= data[i]
  print 'Query point={}, searched dist={}, index={}, data point={}'.format(p_q,d,i,p)

  ax1.cla()
  ax1.scatter(data[:,0], data[:,1], label='data')
  ax1.scatter([p_q[0]], [p_q[1]], marker='x', label='query')
  ax1.scatter([p[0]], [p[1]], marker='+', s=100, color='red', label='searched')
  ax1.legend()
  plt.show(block=False)
  raw_input('enter to continue> ')

  idxes= kdtree.query_ball_point(p_q, 20.0)
  print 'Query point={}, searched indexes={}'.format(p_q,idxes)
  p= data[idxes]

  ax1.cla()
  ax1.scatter(data[:,0], data[:,1], label='data')
  ax1.scatter([p_q[0]], [p_q[1]], marker='x', label='query')
  ax1.scatter(p[:,0], p[:,1], marker='+', s=50, color='red', label='searched')
  ax1.legend()
  plt.show(block=False)
  raw_input('enter to continue> ')
