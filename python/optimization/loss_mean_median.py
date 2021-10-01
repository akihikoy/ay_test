#!/usr/bin/python
#\file    loss_mean_median.py
#\brief   Squared loss vs absolute loss, and mean vs median.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.29, 2021
import scipy.optimize
import numpy as np

if __name__=='__main__':
  D= np.concatenate((np.random.uniform(0,1,size=(10,)), np.random.uniform(0,100,size=(3,)) ))
  #D= np.array([0.,1,3,10])

  f_sqloss= lambda x: np.mean((D-x)**2)
  f_absloss= lambda x: np.mean(np.abs(D-x))

  res_sqloss= scipy.optimize.minimize(f_sqloss,[0.0])
  res_absloss= scipy.optimize.minimize(f_absloss,[0.0])

  print 'D=',D
  print 'mean=',np.mean(D)
  print 'median=',np.median(D)
  print 'res_sqloss=',res_sqloss.x
  print 'res_absloss=',res_absloss.x
