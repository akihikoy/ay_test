#!/usr/bin/python3
#Numerical second order Taylor series expansion.

from taylor_exp_num import *

if __name__=='__main__':
  import math
  #Float version of range
  def FRange1(x1,x2,num_div):
    return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

  #func= lambda x: 0.2*x[0,0]**2 + 0.5*x[1,0]**2 - 0.1*x[0,0]*x[1,0]
  func= lambda x: 0.2*x[0,0]**3 + 0.5*x[1,0]**2 - 0.1*x[0,0]*x[1,0]
  #func= lambda x: math.cos(x[0,0])
  #func= lambda x: math.cos(x[0,0])+math.sin(x[1,0])
  #func= lambda x: 1.0 if abs(x[0,0])<1.0 else 0.0
  #func= lambda x: 1.0 if abs(x[0,0]**2+x[1,0]**2)<2.0 else 0.0

  fp= open('/tmp/true.dat','w')
  for x1 in FRange1(-3.0,3.0,50):
    for x2 in FRange1(-3.0,3.0,50):
      y= func(np.mat([x1,x2]).T)
      fp.write('%f %f %f\n' % (x1,x2,y))
    fp.write('\n')
  fp.close()

  x0= np.mat([1.0,1.0]).T
  a,b,c= TaylorExp2(func, x0)
  print('a,b,c=',a,b,c)

  fp= open('/tmp/approx.dat','w')
  for x1 in FRange1(-3.0,3.0,50):
    for x2 in FRange1(-3.0,3.0,50):
      x= np.mat([x1,x2]).T
      y= ( a + b.T*(x-x0) + 0.5*(x-x0).T*c*(x-x0) )[0,0]
      fp.write('%f %f %f\n' % (x1,x2,y))
    fp.write('\n')
  fp.close()

  print('Plot with')
  print('''qplot -x -3d /tmp/true.dat w l /tmp/approx.dat w l''')
