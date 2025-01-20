#!/usr/bin/python3
#\file    funcs_2d.py
#\brief   Objective functions to be minimized (ver.2, considering a batch operation).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.18, 2021
import numpy as np
import functools
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

def Func(x,kind=0,env=np):
  if kind==0:  return ( x[:,0]+2.0*x[:,1] ).reshape((-1,1))
  if kind==1:  return ( (x[:,0]-0.5)**2 + (x[:,1]-1.0)**2 ).reshape((-1,1))
  if kind==2:  return ( (x[:,0]-0.5)**2 + (x[:,1]-1.0)**2 + env.sin(10.0*x[:,0]) + env.sin(5.0*x[:,1]) ).reshape((-1,1))
  if kind==3:  return ( 3.0+(x[:,0]**2+(env.sin(3.0*x[:,1]))**2) ).reshape((-1,1))
  if kind==4:  return ( env.where((x[:,0]-0.5)**2 + (x[:,1]-1.0)**2 < 0.25, x[:,0]*0.0-1.0, x[:,0]*0.0) ).reshape((-1,1))
  if kind==5:  return ( (x[:,0]-0.5)**2 + (x[:,1]-1.0)**2 + 5.0*env.sin(10.0*x[:,0]) + 5.0*env.sin(5.0*x[:,1]) ).reshape((-1,1))

if __name__=='__main__':
  import sys
  xmin= [-1.,-1.]
  xmax= [2.,3.]
  fkind= int(sys.argv[1]) if len(sys.argv)>1 else 0
  func= functools.partial(Func, kind=fkind)

  fig= plt.figure()
  ax= fig.add_subplot(1,1,1,projection='3d')
  true_x= np.mgrid[xmin[0]:xmax[0]:(xmax[0]-xmin[0])/100, xmin[1]:xmax[1]:(xmax[1]-xmin[1])/100]
  ax.plot_wireframe(true_x[0], true_x[1], func(true_x[:,:,:].reshape(2,-1).T).reshape(true_x.shape[1:]), color='green', linewidth=1, label='true_func')

  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.show()
