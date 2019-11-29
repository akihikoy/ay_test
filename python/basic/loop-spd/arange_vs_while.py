#!/usr/bin/python
import time
import numpy as np

if __name__=='__main__':
  t_last= 10.0
  dt= 0.001

  t0= time.time()
  L= []
  for t in np.arange(0.0,t_last,dt):
    L+= [t]
  print 'arange+for:', (time.time()-t0)*1.0e3

  t0= time.time()
  L= []
  t= 0.0
  while t<t_last:
    L+= [t]
    t+= dt
  print 'while:', (time.time()-t0)*1.0e3

