#!/usr/bin/python
'''
For a function f(x) and a brief state x~N(m,var_x),
we think a function e(m) = E[f(x)].
By using a Taylor series expansion,
  e(m) = E[f(x)] = f(m) + Tr(A var_x)
  where A is: f(x) = (x-m)' A (x-m) + b' (x-m) + c (Taylor expansion around m).
'''

from taylor_exp_num import *

def Expec(f, x, var_x):
  #c,b,A= TaylorExp2(f, x, h=0.01, maxd1=5.0, maxd2=1.0)
  c,b,A= TaylorExp2(f, x, h=0.01, maxd1=2.0, maxd2=2.0)
  #c,b,A= TaylorExp2(f, x, h=0.01, maxd1=100.0, maxd2=10.0)
  return f(x) + (A*var_x).trace()[0,0]

if __name__=='__main__':
  import math
  #Float version of range
  def FRange1(x1,x2,num_div):
    return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

  #func= lambda x: 0.2*x[0,0]**2
  #func= lambda x: 0.4*x[0,0]**3-0.1*x[0,0]**2-3.0*x[0,0]
  #func= lambda x: math.cos(x[0,0])
  func= lambda x: 1.0 if abs(x[0,0])<0.5 else 0.0

  fp= file('/tmp/true.dat','w')
  for xx in FRange1(-3.0,3.0,50):
    y= func(np.mat([xx]))
    fp.write('%f %f\n' % (xx,y))
  fp.close()

  var_x= np.mat([[1.0**2]])

  fp= file('/tmp/expec.dat','w')
  for xx in FRange1(-3.0,3.0,2000):
    x= np.mat([xx]).T
    y= Expec(func, x, var_x)
    fp.write('%f %f\n' % (xx,y))
  fp.close()

  print 'Plot with'
  print '''qplot -x /tmp/true.dat w l /tmp/expec.dat w l'''
