#!/usr/bin/python
#\file    online_linear_reg2.py
#\brief   Online linear regression.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.20, 2017
import numpy as np
import random
import math

#Return row vector of np.matrix.
def MRVec(x):
  if x is None:  return np.mat([])
  elif isinstance(x,list):  return np.mat(x).ravel()
  elif isinstance(x,(np.ndarray,np.matrix)):  return np.mat(x.ravel())
  raise Exception('Len: Impossible to serialize:',x)

#Return column vector of np.matrix.
def MCVec(x):
  return MRVec(x).T

def AddOne(x):
  if isinstance(x,list):
    if len(x)==0:  return [1.0]
    if isinstance(x[0],list): return [x[0]+[1.0]]
    return x+[1.0]
  if isinstance(x,np.ndarray):
    return x.tolist()+[1.0]
  if x.shape[0]==1:  return np.mat(x.tolist()[0]+[1.0])
  if x.shape[1]==1:  return np.mat(x.T.tolist()[0]+[1.0]).T
  return None

def UpdateWeight(W, x, y, alpha=0.1):
  xi= MCVec(AddOne(x))
  y= MCVec(y)
  err= y - W.T.dot(xi)
  W+= alpha*xi.dot(err.T)

def Rand(xmin, xmax):
  return (xmax-xmin)*random.random()+xmin

def Func(x):
  #return [2.0*x[0]+3.0*x[1]-10.0, 3.0-(x[0]*x[0]+x[1]*x[1])]
  return [2.0*x[0]+3.0*x[1]-2.0+0.001*Rand(-1.0,1.0), 3.0-(x[0]*x[0]+x[1]*x[1])+0.001*Rand(-1.0,1.0)]
  #return [x[0]*math.sin(3.0*x[1])]
  #return [x[0]*math.sin(3.0*x[1]), 3.0-(x[0]*x[0]+x[1]*x[1])]
  #return 3.0-(x[0]*x[0]+Sq(math.sin(3.0*x[1])))
  #return 1.0 if x[0]*x[0]+x[1]*x[1] < 0.25 else 0.0

def Main():
  ydim= 2
  xdim= 2
  #xmin= [-0.05,-0.05]
  #xmax= [0.05,0.05]
  xmin= [-0.5,-0.5]
  xmax= [0.5,0.5]

  data_x= []
  data_y= []
  N_sample= 200
  W= np.mat([[0.0 for c in range(ydim)] for r in range(xdim+1)])
  #W= np.mat([[Rand(-0.01,0.01) for c in range(ydim)] for r in range(xdim+1)])
  for i in range(N_sample):
    x= [Rand(xmin[d],xmax[d]) for d in range(xdim)]
    y= Func(x)
    UpdateWeight(W, x, y, alpha=max(0.1,0.5/float(i+1)))
    data_x.append(x)
    data_y.append(y)

  fp= open('res/olr_data.dat','w')
  for x,y in zip(data_x, data_y):
    fp.write('%s %s\n' % (' '.join(map(str,x)),' '.join(map(str,y))))
  fp.close()

  fp= open('res/olr_approx.dat','w')
  for x0 in np.mgrid[xmin[0]:xmax[0]:49j]:
    for x1 in np.mgrid[xmin[1]:xmax[1]:49j]:
      x= [x0,x1]
      y= W.T.dot(MCVec(AddOne(x))).ravel().tolist()[0]
      fp.write('%s %s\n' % (' '.join(map(str,x)),' '.join(map(str,y))))
    fp.write('\n')
  fp.close()

def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa -3d
          res/olr_approx.dat u 1:2:3 w l res/olr_data.dat u 1:2:3 w p
          &''',
    '''qplot -x2 aaa -3d
          res/olr_approx.dat u 1:2:4 w l res/olr_data.dat u 1:2:4 w p
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
