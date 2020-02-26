#!/usr/bin/python
#\file    opt_ctrl1.py
#\brief   Optimal and adaptive control with linear regression and QP
#         Same problem as ../cma_es/opt_ctrl1.py
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.20, 2017
import numpy as np
import random
import math
import scipy.optimize
from online_linear_reg2 import MRVec,MCVec,AddOne,UpdateWeight

#Dynamics (unknown)
def Fdynamics(x):
  f= [0.2*x[0]+0.2*x[1]+0.2, -0.1*x[0]-0.2*x[1]-0.3, 0.5*x[0]-0.1*x[1]+0.2]
  return [fi + 0.2*(random.random()-0.5) for fi in f]

#Objective function to be minimized
def Fcost(f):
  return sum((fi**2 for fi in f))

def Main():
  #options['bounds']= [[-2.0]*2,[2.0]*2]
  #options['parameters0']= [1.0,1.0]

  xdim= 2
  ydim= 3
  W= np.mat([[0.0 for c in range(ydim)] for r in range(xdim+1)])

  N_sample= 1000
  fp= open('res/oc1_lc.dat','w')
  for i in range(N_sample):
    if i<2:
      x= [1.0+0.5*(random.random()-0.5),1.0+0.5*(random.random()-0.5)]
    else:
      #xnew= scipy.optimize.minimize(lambda x:Fcost(W.T.dot(MCVec(AddOne(x))).ravel().tolist()[0]), x).x
      WWT= W.dot(W.T)
      xnew= (-WWT[:xdim,:xdim].I*WWT[:xdim,xdim]).ravel().tolist()[0]
      d= math.sqrt(sum(((x1-x0)**2 for x1,x0 in zip(xnew,x))))
      dmax= 0.5
      if d>dmax:  x= [x0+(dmax/d)*(x1-x0) for x1,x0 in zip(xnew,x)]
      else:       x= xnew
    f= Fdynamics(x)
    #Learning Dynamics
    UpdateWeight(W, x, f, alpha=max(0.1,0.5/float(i+1)))
    c= Fcost(f)
    fp.write('%f %f %f\n'%(x[0],x[1],c))
  fp.close

  fp= open('res/oc1_true.dat','w')
  #X= np.mgrid[-2:2:0.1, -2:2:0.1]
  #for x in zip(X[0].ravel(),X[1].ravel()):
  X= np.ogrid[-2:2:0.1, -2:2:0.1]
  for x0 in X[0].ravel():
    for x1 in X[1].ravel():
      x= [x0,x1]
      c= Fcost(Fdynamics(x))
      fp.write('%f %f %f\n'%(x[0],x[1],c))
    fp.write('\n')
  fp.close

  fp= open('res/oc1_approx.dat','w')
  X= np.ogrid[-2:2:0.1, -2:2:0.1]
  for x0 in X[0].ravel():
    for x1 in X[1].ravel():
      x= [x0,x1]
      c= Fcost(W.T.dot(MCVec(AddOne(x))).ravel().tolist()[0])
      fp.write('%f %f %f\n'%(x[0],x[1],c))
    fp.write('\n')
  fp.close

def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa -3d
          res/oc1_approx.dat w l
          res/oc1_true.dat w l
          res/oc1_lc.dat w lp
          &''',
    '''qplot -x2 aaa
          res/oc1_lc.dat u 3 w l
          &''',
    '''''',
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.splitlines())
      print '###',cmd
      os.system(cmd)

  print '##########################'
  print '###Press enter to close###'
  print '##########################'
  raw_input()
  os.system('qplot -x2kill aaa')

if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()
