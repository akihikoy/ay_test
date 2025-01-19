#!/usr/bin/python3
#\file    opt_ctrl1.py
#\brief   Optimal and adaptive control with CMA-ES
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.20, 2017
from optimizers1 import *

#Dynamics (unknown)
def Fdynamics(x):
  f= [0.2*x[0]+0.2*x[1]+0.2, -0.1*x[0]-0.2*x[1]-0.3, 0.5*x[0]-0.1*x[1]+0.2]
  return [fi + 0.2*(random.random()-0.5) for fi in f]

#Objective function to be minimized
def Fcost(f):
  return sum((fi**2 for fi in f))

def Main():
  opt= TContOptNoGrad()
  options= {}
  options['bounds']= [[-2.0]*2,[2.0]*2]
  options['tolfun']= 1.0e-4
  options['scale0']= 0.05
  #options['popsize']= 3
  options['parameters0']= [1.0,1.0]
  options['maxfevals']= 1000
  opt.Init({'options':options})

  fp= open('/tmp/cma1_lc.dat','w')
  while not opt.Stopped():
    x= opt.Select()
    f= Fcost(Fdynamics(x))
    opt.Update(-f)
    fp.write('%f %f %f\n'%(x[0],x[1],f))
  fp.close
  print('Result=',opt.Result())

  fp= open('/tmp/cma1_true.dat','w')
  #X= np.mgrid[-2:2:0.1, -2:2:0.1]
  #for x in zip(X[0].ravel(),X[1].ravel()):
  X= np.ogrid[-2:2:0.1, -2:2:0.1]
  for x0 in X[0].ravel():
    for x1 in X[1].ravel():
      x= [x0,x1]
      f= Fcost(Fdynamics(x))
      fp.write('%f %f %f\n'%(x[0],x[1],f))
    fp.write('\n')
  fp.close

def PlotGraphs():
  print('Plotting graphs..')
  import os
  commands=[
    '''qplot -x2 aaa -3d
          /tmp/cma1_true.dat w l
          /tmp/cma1_lc.dat w lp
          &''',
    '''qplot -x2 aaa
          /tmp/cma1_lc.dat u 3 w l
          &''',
    '''''',
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.splitlines())
      print('###',cmd)
      os.system(cmd)

  print('##########################')
  print('###Press enter to close###')
  print('##########################')
  input()
  os.system('qplot -x2kill aaa')

if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()
