#!/usr/bin/python3
#\file    grad_ascent2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.21, 2016

from grad_ascent import *
from taylor_exp.taylor_exp_num import TaylorExp1
import random

#Generate a random number of uniform distribution of specified bound.
def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

#Generate a random vector of uniform distribution; each dim has different bound.
def RandN(xmins,xmaxs):
  assert(len(xmins)==len(xmaxs))
  return [Rand(xmins[d],xmaxs[d]) for d in range(len(xmins))]

#Generate a random vector of uniform distribution; each dim has different bound.
def RandB(bounds):
  return RandN(bounds[0],bounds[1])

#Convert a np.ndarray or single row/column np.matrix to a list.
def ToList(x):
  if x is None:  return []
  elif isinstance(x,list):  return x
  elif isinstance(x,(np.ndarray,np.matrix)):
    if len(x.shape)==1:  return x.tolist()
    if len(x.shape)==2:
      if x.shape[0]==1:  return x.tolist()[0]
      if x.shape[1]==1:  return x.T.tolist()[0]
      if x.shape[0]==0 and x.shape[1]==0:  return []
  raise Exception('ToList: Impossible to serialize:',x)

Bounds= [[0.2,0.3],[0.8,0.8]]  #Boundary of y,z

def Func(x):
  x= ToList(x)
  #print 'x', x
  y,z= x[0],x[1]
  #Dynamics(y,z)
  cy= y + 0.1*z**0.5
  cz= 0.0
  w=  0.4*z
  #return cy,cz,w
  yl= 0.5  #Location of receiving container.
  wl= 0.3  #Size of receiving container
  #Assess(cy,cz,w):
  if (cy+0.5*w) >= (yl+0.5*wl):
    return (yl+0.5*wl) - (cy+0.5*w)  #Penalty
    #return -1
  if (cy-0.5*w) <= (yl-0.5*wl):
    return (cy-0.5*w) - (yl-0.5*wl)  #Penalty
    #return -1
  e= +1.0 - 50.0*(cy-yl)**2
  return e if e>0.0 else e

def Grad(x):
  y,dy= TaylorExp1(Func, x)
  #print 'dy',np.array(ToList(dy))
  return np.array(ToList(dy))

def Cnstr(x):
  return np.array([max(xmin,min(xmax,xe)) for xe,xmin,xmax in zip(x,Bounds[0],Bounds[1])])


def Main():
  nt= 25
  x_true= np.array(sum([[[x1,x2] for x2 in FRange1(Bounds[0][0],Bounds[1][0],nt)] for x1 in FRange1(Bounds[0][1],Bounds[1][1],nt)],[]))
  y_true= np.array([[Func(x)] for x in x_true])
  y_grad= np.array([Grad(x) for x in x_true])

  fp1= open('/tmp/true.dat','w')
  for x,y,i in zip(x_true,y_true,list(range(len(y_true)))):
    if i%(nt+1)==0:  fp1.write('\n')
    fp1.write('%s %s\n' % (' '.join(map(str,x)),' '.join(map(str,y))))
  fp1.close()

  fp1= open('/tmp/grad.dat','w')
  for x,y,g,i in zip(x_true,y_true,y_grad,list(range(len(y_grad)))):
    if i%(nt+1)==0:  fp1.write('\n')
    fp1.write('%s %s %s\n' % (' '.join(map(str,x)),' '.join(map(str,y)),' '.join(map(str,g)) ))
  fp1.close()

  x0= RandB(Bounds)
  print('x0',x0)

  fp1= open('/tmp/opt_gd.dat','w')
  opt= TGradientAscent(alpha=0.005, normalize_grad=True)
  RunFirstOpt(opt, Func,Grad,x0, f_cnstr=Cnstr, fp=fp1, n_iter=200)
  fp1.close()

  print('-----')

  fp1= open('/tmp/opt_adadelta.dat','w')
  opt= TAdaDeltaMax(rho=0.98, eps=1.0e-6)
  RunFirstOpt(opt, Func,Grad,x0, f_cnstr=Cnstr, fp=fp1, n_iter=200)
  fp1.close()


def PlotGraphs():
  print('Plotting graphs..')
  import os
  qopt= ''
  #qopt= '-o -.svg'
  commands=[
    '''qplot -x2 aaa -3d -s 'set xlabel "x";set ylabel "y";set zlabel "f";set ticslevel 0;'
          /tmp/true.dat u 1:2:3         w l  lw 1 lt 2 t '"true"'
          /tmp/opt_gd.dat u 1:2:3       w lp lw 3 lt 1 t '"gradient ascent"'
          /tmp/opt_adadelta.dat u 1:2:3 w lp lw 3 lt 3 t '"AdaDelta"'
        &''',
          #-o res/xyf.jpg
          #/tmp/grad.dat u 1:2:3:'(0.05*$4):(0.05*$5):(0.05)' w vector lw 2 t '"grad"'
    '''qplot -x2 aaa -s 'set xlabel "step";set ylabel "f";set key right top;set logscale x;'
          /tmp/opt_gd.dat u 3       w lp lw 2 lt 1 t '"gradient ascent"'
          /tmp/opt_adadelta.dat u 3 w lp lw 2 lt 3 t '"AdaDelta"'
        &''',
          #-o res/learning_curve.jpg
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.splitlines())
      if qopt!='':
        cmd= cmd.replace('qplot -x2 aaa','qplot '+qopt)
        if cmd[-1]=='&':  cmd= cmd[:-1]
      print('###',cmd)
      os.system(cmd)

  print('##########################')
  print('###Press enter to close###')
  print('##########################')
  input()
  if qopt=='':  os.system('qplot -x2kill aaa')


if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()
