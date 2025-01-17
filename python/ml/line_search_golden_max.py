#!/usr/bin/python3
'''Implementation of a line search algorithm, Golden search'''
import numpy as np
import numpy.linalg as la
import math

golden_ratio= (math.sqrt(5.0)-1.0)*0.5
def LineSearchGolden(f,x0,direc,l_max=1.0,n_max=5,tol=1.0e-3):
  a= np.mat(x0)
  b= np.mat(x0)+np.mat(direc)*l_max

  c= b - golden_ratio*(b-a)
  d= a + golden_ratio*(b-a)
  fc= f(c); fd= f(d)
  while la.norm(c-d)>tol and n_max>0:
    if fc>fd:
      b= d
      d= c
      c= b - golden_ratio*(b-a)
      fd=fc; fc=f(c)
    else:
      a= c
      c= d
      d= a + golden_ratio*(b-a)
      fc=fd; fd=f(d)
    n_max-= 1
  return (b+a)*0.5


if __name__=='__main__':
  def PrintEq(s):  print('%s= %r' % (s, eval(s)))
  def FRange1(x1,x2,num_div):
    return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

  from taylor_exp.taylor_exp_num import TaylorExp2

  f= lambda x: -0.2*(x[0,0]-1.0)**2
  #f= lambda x: -0.2*(x[0,0]-1.0)**2 - 1.5*x[0,0]
  #f= lambda x: math.sin(x[0,0])

  fp= open('/tmp/opt.dat','w')
  x= np.mat([-1.5])
  fp.write('%f %f\n' % (x[0,0],f(x)))
  for i in range(10):
    f0,grad,H= TaylorExp2(f,x)
    x= LineSearchGolden(f,x0=x,direc=grad,l_max=1.0,tol=1.0e-3)
    print(x, f(x))
    fp.write('%f %f\n' % (x[0,0],f(x)))
  fp.close()

  fp= open('/tmp/true.dat','w')
  for xx in FRange1(-5.0,5.0,200):
    y= f(np.mat([xx]))
    fp.write('%f %f\n' % (xx,y))
  fp.close()

  print('''qplot -x /tmp/true.dat w l /tmp/opt.dat w p lt 3''')

