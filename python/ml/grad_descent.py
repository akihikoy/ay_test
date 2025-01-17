#!/usr/bin/python3
#\file    grad_descent.py
#\brief   Gradient descent algorithms (e.g. Ada-Delta).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.26, 2015
import math
import numpy as np
import numpy.linalg as la

def GradientDescent(f_obj, f_grad, x0, f_cnstr=None, n_iter=20, alpha=0.2, fp=None):
  x= x0
  y= f_obj(x)
  print(x, y)
  if fp:  fp.write('%s %s\n' % (' '.join(map(str,x)),str(y)))
  for i in range(n_iter):
    g= f_grad(x)
    x= x - alpha*g
    if f_cnstr:  x= f_cnstr(x)
    y= f_obj(x)
    print(x, y, g)
    if fp:  fp.write('%s %s\n' % (' '.join(map(str,x)),str(y)))


#ref. http://qiita.com/skitaoka/items/e6afbe238cd69c899b2a
# http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
def StepAdaDelta(x, g, r, s, rho, eps):
  r= rho*r + (1.0-rho)*g*g
  v= math.sqrt((s+eps)/(r+eps)) * g
  s= rho*s + (1.0-rho)*v*v
  x= x - v
  return x, r, s

StepAdaDeltaV= np.vectorize(StepAdaDelta)

def AdaDelta(f_obj, f_grad, x0, f_cnstr=None, n_iter=20, rho=0.95, eps=1.0e-3, fp=None):
  x= x0
  y= f_obj(x)
  print(x, y)
  if fp:  fp.write('%s %s\n' % (' '.join(map(str,x)),str(y)))
  r= np.array([0.0]*2)
  s= np.array([0.0]*2)
  for i in range(n_iter):
    g= f_grad(x)
    x,r,s= StepAdaDeltaV(x,g,r,s, rho,eps)
    if f_cnstr:  x= f_cnstr(x)
    y= f_obj(x)
    print(x, y, g)
    if fp:  fp.write('%s %s\n' % (' '.join(map(str,x)),str(y)))


#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Main():
  S= math.sin
  C= math.cos
  Func= lambda x: S(2.0*x[0])*x[0] + C(x[1]*x[1])*x[1]
  Grad= lambda x: np.array([2.0*C(2.0*x[0])*x[0]+S(2.0*x[0]),
                            -2.0*S(x[1]*x[1])*x[1]*x[1]+C(x[1]*x[1])])
  Cnstr= lambda x: np.array([max(-1.0,min(1.0,xe)) for xe in x])

  nt= 25
  x_true= np.array(sum([[[x1,x2] for x2 in FRange1(-1.0,1.0,nt)] for x1 in FRange1(-1.0,1.0,nt)],[]))
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

  '''NOTE:
  Saddle point of Func is [0, 0.80825193293576669], obtained by:
  >>> import scipy.optimize as opt
  >>> import math
  >>> opt.broyden1(lambda x: -2.0*math.sin(x[0]*x[0])*x[0]*x[0]+math.cos(x[0]*x[0]), [0.8], f_tol=1e-16)[0]
  0.80825193293576669
  '''

  #x0= [-0.5,0.8082]
  x0= [-1.0,0.8082]

  fp1= open('/tmp/opt_gd.dat','w')
  GradientDescent(Func,Grad,x0, f_cnstr=Cnstr, fp=fp1)
  fp1.close()

  print('')

  fp1= open('/tmp/opt_adadelta.dat','w')
  AdaDelta(Func,Grad,x0, f_cnstr=Cnstr, fp=fp1)
  fp1.close()


def PlotGraphs():
  print('Plotting graphs..')
  import os
  commands=[
    '''qplot -x2 aaa -3d -s 'set xlabel "x";set ylabel "y";set zlabel "f";set ticslevel 0;'
          /tmp/true.dat u 1:2:3         w l  lw 1 lt 2 t '"true"'
          /tmp/opt_gd.dat u 1:2:3       w lp lw 3 lt 1 t '"gradient descent"'
          /tmp/opt_adadelta.dat u 1:2:3 w lp lw 3 lt 3 t '"AdaDelta"'
        &''',
          #-o res/xyf.jpg
          #/tmp/grad.dat u 1:2:3:'(0.05*$4):(0.05*$5):(0.05)' w vector lw 2 t '"grad"'
    '''qplot -x2 aaa -s 'set xlabel "step";set ylabel "f";set key right top;'
          /tmp/opt_gd.dat u 3       w lp lw 2 lt 1 t '"gradient descent"'
          /tmp/opt_adadelta.dat u 3 w lp lw 2 lt 3 t '"AdaDelta"'
        &''',
          #-o res/learning_curve.jpg
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
