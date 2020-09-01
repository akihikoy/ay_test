#!/usr/bin/python
#\file    funcs.py
#\brief   Objective functions to be minimized.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.11, 2020
import numpy as np

def Func(x,kind=0):
  if kind==0:  return x[0]+2.0*x[1]
  if kind==1:  return (x[0]-0.5)**2 + (x[1]-1.0)**2
  if kind==2:  return (x[0]-0.5)**2 + (x[1]-1.0)**2 + np.sin(10.0*x[0]) + np.sin(5.0*x[1])
  if kind==3:  return 3.0+(x[0]*x[0]+(np.sin(3.0*x[1]))**2)
  if kind==4:  return -1.0 if (x[0]-0.5)**2 + (x[1]-1.0)**2 < 0.25 else 0.0
  if kind==5:  return (x[0]-0.5)**2 + (x[1]-1.0)**2 + 5.0*np.sin(10.0*x[0]) + 5.0*np.sin(5.0*x[1])

def Plot(xmin,xmax,func,x_points=None,N=50):
  with open('/tmp/func.dat','w') as fp:
    for x0 in np.arange(xmin[0],xmax[0],(xmax[0]-xmin[0])/N):
      for x1 in np.arange(xmin[1],xmax[1],(xmax[1]-xmin[1])/N):
        x= [x0,x1]
        fp.write('%f %f %f\n' % (x0,x1,func(x)))
      fp.write('\n')

  if x_points is not None:
    with open('/tmp/points.dat','w') as fp:
      for x in x_points:
        fp.write('%f %f %f\n' % (x[0],x[1],func(x)))
    print 'Plot by:'
    print '  qplot -x -3d /tmp/func.dat w l /tmp/points.dat'
  else:
    print 'Plot by:'
    print '  qplot -x -3d /tmp/func.dat w l'

if __name__=='__main__':
  import sys
  xmin= [-1.,-1.]
  xmax= [2.,3.]
  fkind= int(sys.argv[1]) if len(sys.argv)>1 else 0
  Plot(xmin,xmax,lambda x:Func(x,fkind),x_points=[[0.0,0.0]])
