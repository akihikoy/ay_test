#!/usr/bin/python3
#Numerical second order Taylor series expansion.
import numpy as np

#Return a vector [x_0,...,x_D-1], x_d=0 (d!=i), x_i=h
def Delta1(D,i,h):
  delta= np.mat([0.0]*D).T
  delta[i]= h
  return delta

#Return a vector [x_0,...,x_D-1], x_d=0 (d!=i1,i2), x_i1=h1, x_i2=h2
def Delta2(D,i1,i2,h1,h2):
  delta= np.mat([0.0]*D).T
  if i1==i2:
    delta[i1]= h1+h2
  else:
    delta[i1]= h1
    delta[i2]= h2
  return delta

'''First order Taylor series expansion of f around x0.
  f(x) ~ a  +  b.T * (x-x0)
Returns a,b.
h: Used for numerical derivative computation.
'''
def TaylorExp1(f, x0, h=0.01, maxd1=5.0):
  x0= np.mat(x0)
  if x0.shape[0]==1:  x0= x0.T
  Dx= x0.shape[0]
  a= f(x0)
  b= np.mat([0.0]*Dx).T
  maxb= 0.0
  for d in range(Dx):
    b[d]= ( f(x0+Delta1(Dx,d,h)) - f(x0-Delta1(Dx,d,h)) ) / (2.0*h)
    if abs(b[d])>maxb:  maxb= abs(b[d])
  if maxb>maxd1:  b*= maxd1/maxb
  return a,b

'''Second order Taylor series expansion of f around x0.
  f(x) ~ a  +  b.T * (x-x0)  +  1/2 * (x-x0).T * c * (x-x0)
Returns a,b,c.
h: Used for numerical derivative computation.
'''
def TaylorExp2(f, x0, h=0.01, maxd1=5.0, maxd2=1.0):
  x0= np.mat(x0)
  if x0.shape[0]==1:  x0= x0.T
  Dx= x0.shape[0]
  a= f(x0)
  b= np.mat([0.0]*Dx).T
  maxb= 0.0
  for d in range(Dx):
    b[d]= ( f(x0+Delta1(Dx,d,h)) - f(x0-Delta1(Dx,d,h)) ) / (2.0*h)
    if abs(b[d])>maxb:  maxb= abs(b[d])
  if maxb>maxd1:  b*= maxd1/maxb
  h= 0.5*h
  c= np.mat([[0.0]*Dx]*Dx)
  maxc= 0.0
  for d1 in range(Dx):
    c[d1,d1]= ( f(x0+Delta1(Dx,d1,2.0*h)) - 2.0*f(x0) + f(x0-Delta1(Dx,d1,2.0*h)) ) / (4.0*h**2)
    if abs(c[d1,d1])>maxc:  maxc= abs(c[d1,d1])
    for d2 in range(d1+1,Dx):
      c[d1,d2]= ( f(x0+Delta2(Dx,d1,d2,h,h)) - f(x0+Delta2(Dx,d1,d2,h,-h)) - f(x0+Delta2(Dx,d1,d2,-h,h)) + f(x0-Delta2(Dx,d1,d2,h,h)) ) / (4.0*h**2)
      c[d2,d1]= c[d1,d2]
      if abs(c[d1,d2])>maxc:  maxc= abs(c[d1,d2])
  if maxc>maxd2:  c*= maxd2/maxc
  return a,b,c


if __name__=='__main__':
  import math
  #Float version of range
  def FRange1(x1,x2,num_div):
    return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

  #func= lambda x: 0.2*x[0,0]**2
  #func= lambda x: x[0,0]**3
  func= lambda x: math.cos(x[0,0])
  #func= lambda x: 1.0 if abs(x[0,0])<0.5 else 0.0

  fp= file('/tmp/true.dat','w')
  for xx in FRange1(-3.0,3.0,50):
    y= func(np.mat([xx]))
    fp.write('%f %f\n' % (xx,y))
  fp.close()

  x0= np.mat([0.5])
  a,b,c= TaylorExp2(func, x0)
  print('a,b,c=',a,b,c)

  fp= file('/tmp/approx.dat','w')
  for xx in FRange1(-3.0,3.0,50):
    x= np.mat([xx]).T
    y= ( a + b.T*(x-x0) + 0.5*(x-x0).T*c*(x-x0) )[0,0]
    fp.write('%f %f\n' % (xx,y))
  fp.close()

  print('Plot with')
  print('''qplot -x /tmp/true.dat w l /tmp/approx.dat w l''')
