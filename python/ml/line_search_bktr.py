#!/usr/bin/python3
'''Implementation of a line search algorithm, Backtracking line search'''
import numpy as np
import numpy.linalg as la
import math

def LineSearchBkTr(f,x0,direc,grad=None,l0=1.0,rho=0.5,eps=0.5,f0=None):
  d= np.mat(direc)
  if grad!=None:  g= np.mat(grad)
  else:           g= -d
  t= -eps * (g.T * d)[0,0]
  x= np.mat(x0)
  l= l0
  if f0==None:  f0= f(x)
  while True:
    x2= x+l*d
    if f0 - f(x2) >= l*t:  return x2
    l*= rho


if __name__=='__main__':
  def PrintEq(s):  print('%s= %r' % (s, eval(s)))
  def FRange1(x1,x2,num_div):
    return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

  from taylor_exp.taylor_exp_num import TaylorExp2

  #f= lambda x: 0.2*(x[0,0]-1.0)**2
  f= lambda x: 0.2*(x[0,0]-1.0)**2 + 1.5*x[0,0]
  #f= lambda x: math.sin(x[0,0])

  fp= open('/tmp/opt.dat','w')
  x= np.mat([1.5])
  fp.write('%f %f\n' % (x[0,0],f(x)))
  for i in range(10):
    f0,grad,H= TaylorExp2(f,x)
    x= LineSearchBkTr(f,x0=x,direc=-grad,grad=grad,l0=1.0,rho=0.5,eps=0.5,f0=f0)
    print(x, f(x))
    fp.write('%f %f\n' % (x[0,0],f(x)))
  fp.close()

  fp= open('/tmp/true.dat','w')
  for xx in FRange1(-5.0,5.0,200):
    y= f(np.mat([xx]))
    fp.write('%f %f\n' % (xx,y))
  fp.close()

  print('''qplot -x /tmp/true.dat w l /tmp/opt.dat w p lt 3''')
