#!/usr/bin/python
#\file    num_expec.py
#\brief   Numerical expectation
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.17, 2015

import math
import numpy as np

def Len(x):
  if x==None:  return 0
  elif isinstance(x,list):  return len(x)
  elif isinstance(x,(np.ndarray,np.matrix)):
    if len(x.shape)==1:  return x.shape[0]
    if len(x.shape)==2:
      if x.shape[0]==1:  return x.shape[1]
      if x.shape[1]==1:  return x.shape[0]
      if x.shape[0]==0 or x.shape[1]==0:  return 0
  raise Exception('Len: Impossible to serialize:',x)

#Return a vector [x_0,...,x_D-1], x_d=0 (d!=i), x_i=h
def Delta1(D,i,h):
  delta= np.mat([[0.0]]*D)
  delta[i,0]= h
  return delta

#Return a vector [x_0,...,x_D-1], x_d=0 (d!=i1,i2), x_i1=h1, x_i2=h2
def Delta2(D,i1,i2,h1,h2):
  delta= np.mat([[0.0]]*D)
  if i1==i2:
    delta[i1,0]= h1+h2
  else:
    delta[i1,0]= h1
    delta[i2,0]= h2
  return delta

def iGaussian(dx, invSigma):
  return math.exp(-0.5*(dx.T*invSigma*dx)[0,0])

#Get expectation of f numerically w.r.t. Gaussian(x0,var)
def NumExpec(f, x0, var, nsd=(0.25,0.5,1.0)):
  invSigma= var.I
  Dx= Len(x0)
  sd= [math.sqrt(var[d,d]) for d in range(Dx)]
  #integ= np.mat([0.0]*Dx).T
  #sumG= 0.0
  integ= f(x0)
  sumG= 1.0
  for h in nsd:
    for d1 in range(Dx):
      h1= h*sd[d1]
      dx=Delta1(Dx,d1,h1)  ; g=iGaussian(dx, invSigma); integ+=f(x0+dx)*g; sumG+=g
      dx=Delta1(Dx,d1,-h1) ; g=iGaussian(dx, invSigma); integ+=f(x0+dx)*g; sumG+=g
      for d2 in range(d1+1,Dx):
        h2= h*sd[d2]
        dx=Delta2(Dx,d1,d2,h1,h2)    ; g=iGaussian(dx, invSigma); integ+=f(x0+dx)*g; sumG+=g
        dx=Delta2(Dx,d1,d2,-h1,h2)   ; g=iGaussian(dx, invSigma); integ+=f(x0+dx)*g; sumG+=g
        dx=Delta2(Dx,d1,d2,h1,-h2)   ; g=iGaussian(dx, invSigma); integ+=f(x0+dx)*g; sumG+=g
        dx=Delta2(Dx,d1,d2,-h1,-h2)  ; g=iGaussian(dx, invSigma); integ+=f(x0+dx)*g; sumG+=g
  return integ/sumG


#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Main():
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))
  import math

  #func= lambda x: 0.2*x[0,0]**2 + 0.5*x[1,0]**2 - 0.1*x[0,0]*x[1,0]
  #func= lambda x: 0.2*x[0,0]**3 + 0.5*x[1,0]**2 - 0.1*x[0,0]*x[1,0]
  #func= lambda x: math.cos(x[0,0])
  #func= lambda x: math.cos(x[0,0])+math.sin(x[1,0])
  #func= lambda x: 1.0 if abs(x[0,0])<1.0 else 0.0
  #func= lambda x: 1.0 if abs(x[0,0]**2+x[1,0]**2)<2.0 else 0.0
  func= lambda x: (1.0 if abs(max(x[0,0],x[1,0]))<1.0 else 0.0)*(1.0 if x[0,0]>0.0 else -1.0)

  fp= file('/tmp/true.dat','w')
  for x1 in FRange1(-3.0,3.0,50):
    for x2 in FRange1(-3.0,3.0,50):
      y= func(np.mat([x1,x2]).T)
      fp.write('%f %f %f\n' % (x1,x2,y))
    fp.write('\n')
  fp.close()

  var= np.mat([[1.0, 0.0],
               [0.0, 0.5]])
  fp= file('/tmp/expec.dat','w')
  for x1 in FRange1(-3.0,3.0,50):
    for x2 in FRange1(-3.0,3.0,50):
      x= np.mat([x1,x2]).T
      y= NumExpec(f=lambda x: np.mat([func(x)]), x0=x, var=var)
      fp.write('%f %f %f\n' % (x1,x2,y))
    fp.write('\n')
  fp.close()



def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa -3d /tmp/true.dat w l /tmp/expec.dat w l &''',
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
