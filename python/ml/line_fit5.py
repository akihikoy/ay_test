#!/usr/bin/python
#\file    line_fit5.py
#\brief   polyfit test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.16, 2024
import numpy as np
import numpy.linalg as la
import math
import random
import time
import os,sys
import matplotlib.pyplot as plt

def LineFit(data_x, data_f):
  w= np.polyfit(data_x, data_f, 1)
  return w

if __name__=='__main__':
  data_id= int(sys.argv[1]) if len(sys.argv)>1 else 0
  markers= ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
  fig= plt.figure(figsize=(5,5))
  ax= fig.add_subplot(1,1,1,title='line_fit5(data_id={})'.format(data_id),xlabel='time',ylabel='center-y')
  x_base= None
  for idx in range(0,1000,3):
    filename= 'data/obj_tr/obj_tr_{}_{}.dat'.format(data_id,idx)
    if not os.path.exists(filename):  break
    with open(filename) as fp:
      d= fp.read()
      a= np.array([map(float,s.split()) for s in d.split('\n') if s!=''])
      data_x= a[:,0]
      data_f= a[:,1]
    if x_base is None:  x_base= np.min(data_x)
    data_x= data_x-x_base
    xmin,xmax= np.min(data_x),np.max(data_x)
    w= LineFit(data_x, data_f)
    #print 'Computation time[ms]:',(time.time()-t0)*1000.
    print 'idx={}, n={}, x=[{:.2f},{:.2f}], w={}'.format(idx,len(data_x),xmin,xmax,w)
    test_x= np.linspace(xmin,xmax,50.0)
    test_f= np.dot(np.vstack((test_x,np.ones_like(test_x))).T,w)
    ax.scatter(data_x, data_f, s=0.5)
    ax.plot(data_x, data_f, linewidth=0.1)
    ax.plot(test_x, test_f, linewidth=0.1, color='blue')  #label='fit-{}'.format(idx)
  ax.scatter(data_x, data_f, color='red', s=2)
  #ax.legend()
  plt.show()
