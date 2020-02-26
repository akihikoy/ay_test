#!/usr/bin/python
#\file    quad_approx.py
#\brief   Approximate a quadratic form of a given function at a point (i.e. 2nd order Taylor series expansion) using least squares.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.15, 2015

import numpy as np
import numpy.linalg as la

def ToList(x):
  if x==None:  return []
  elif isinstance(x,list):  return x
  elif isinstance(x,(np.ndarray,np.matrix)):
    if len(x.shape)==1:  return x.tolist()
    if len(x.shape)==2:
      if x.shape[0]==1:  return x.tolist()[0]
      if x.shape[1]==1:  return x.T.tolist()[0]
      if x.shape[0]==0 and x.shape[1]==0:  return []
  raise Exception('ToList: Impossible to serialize:',x)

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
  delta[i]= h
  return delta

#Return a vector [x_0,...,x_D-1], x_d=0 (d!=i1,i2), x_i1=h1, x_i2=h2
def Delta2(D,i1,i2,h1,h2):
  delta= np.mat([[0.0]]*D)
  if i1==i2:
    delta[i1]= h1+h2
  else:
    delta[i1]= h1
    delta[i2]= h2
  return delta

#Get a quadratic feature vector of x.
#x should be a list.
#e.g. for x=[x1,x2], return [x1**2, 2x1x2, x2**2,  x1, x2]
def XToQuadFeat(x):
  Dx= len(x)
  xf= [0.0]*(Dx*(Dx+1)/2 + Dx)
  i= 0
  for d1 in range(Dx):
    xf[i]= x[d1]*x[d1]
    i+= 1
    for d2 in range(d1+1,Dx):
      xf[i]= 2.0*x[d1]*x[d2]
      i+= 1
  for d1 in range(Dx):
    xf[i]= x[d1]
    i+= 1
  return xf

#x0 should be a np.matrix.
def QuadForm(f, x0, h=0.01, f_reg=1.0e-8):
  c= f(x0)
  h2= 0.5*h
  if x0.shape[0]==1:  x0= x0.T
  Dx= x0.shape[0]
  Df= (Dx*(Dx+1)/2 + Dx)
  N= (2*Dx*Dx) * 2  #Num of samples
  V= np.mat([[0.0]]*N)
  Theta= np.mat([[0.0]*Df]*N)
  n= 0
  for d1 in range(Dx):
    x=Delta1(Dx,d1,h)  ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
    x=Delta1(Dx,d1,-h) ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
    x=Delta1(Dx,d1,h2) ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
    x=Delta1(Dx,d1,-h2); V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
    for d2 in range(d1+1,Dx):
      x=Delta2(Dx,d1,d2,h,h)    ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,-h,h)   ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,h,-h)   ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,-h,-h)  ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,h2,h2)  ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,-h2,h2) ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,h2,-h2) ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,-h2,-h2); V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
  #Theta/= h
  #print 'Theta.T*Theta',Theta.T*Theta
  w= la.inv(Theta.T*Theta+f_reg*np.eye(Df))*Theta.T*V
  #print V-Theta*w
  return w,c


#x0 should be a np.matrix.
def QuadForm2(f, x0, h=0.01, f_reg=1.0e-8):
  c= f(x0)
  h2= 0.25*h
  h3= 0.5*h
  h4= 0.75*h
  if x0.shape[0]==1:  x0= x0.T
  Dx= x0.shape[0]
  Df= (Dx*(Dx+1)/2 + Dx)
  N= (2*Dx*Dx) * 4  #Num of samples
  V= np.mat([[0.0]]*N)
  Theta= np.mat([[0.0]*Df]*N)
  n= 0
  for d1 in range(Dx):
    x=Delta1(Dx,d1,h)  ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
    x=Delta1(Dx,d1,-h) ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
    x=Delta1(Dx,d1,h2) ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
    x=Delta1(Dx,d1,-h2); V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
    x=Delta1(Dx,d1,h3) ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
    x=Delta1(Dx,d1,-h3); V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
    x=Delta1(Dx,d1,h4) ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
    x=Delta1(Dx,d1,-h4); V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
    for d2 in range(d1+1,Dx):
      x=Delta2(Dx,d1,d2,h,h)    ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,-h,h)   ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,h,-h)   ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,-h,-h)  ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,h2,h2)  ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,-h2,h2) ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,h2,-h2) ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,-h2,-h2); V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,h3,h3)  ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,-h3,h3) ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,h3,-h3) ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,-h3,-h3); V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,h4,h4)  ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,-h4,h4) ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,h4,-h4) ; V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
      x=Delta2(Dx,d1,d2,-h4,-h4); V[n,0]=f(x0+x)-c; Theta[n,:]=XToQuadFeat(x.T.tolist()[0]); n+=1
  #Theta/= h
  #print 'Theta.T*Theta',Theta.T*Theta
  w= la.inv(Theta.T*Theta+f_reg*np.eye(Df))*Theta.T*V
  #print V-Theta*w
  return w,c

#x0 should be a np.matrix.
def QuadForm3(f, x0, h=0.01, f_reg=1.0e-8, nx=4):
  c= f(x0)
  if x0.shape[0]==1:  x0= x0.T
  Dx= x0.shape[0]
  Df= (Dx*(Dx+1)/2 + Dx)
  N= (2*Dx*Dx) * nx  #Num of samples
  V= np.mat([[0.0]]*N)
  Theta= np.mat([[0.0]*Df]*N)
  n= 0
  for n in range(N):
    x= np.mat((np.random.random(Dx)-0.5)*2.0*h).T
    V[n,0]=f(x0+x)-c
    Theta[n,:]=XToQuadFeat(x.T.tolist()[0])
  #Theta/= h
  #print 'Theta.T*Theta',Theta.T*Theta
  w= la.inv(Theta.T*Theta+f_reg*np.eye(Df))*Theta.T*V
  #print V-Theta*w
  return w,c

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Main():
  def PrintEq(s):  print '%s= %r' % (s, eval(s))
  import math

  func= lambda x: 0.2*x[0,0]**2 + 0.5*x[1,0]**2 - 0.1*x[0,0]*x[1,0]
  #func= lambda x: 0.2*x[0,0]**3 + 0.5*x[1,0]**2 - 0.1*x[0,0]*x[1,0]
  #func= lambda x: math.cos(x[0,0])
  #func= lambda x: math.cos(x[0,0])+math.sin(x[1,0])
  #func= lambda x: 1.0 if abs(x[0,0])<1.0 else 0.0
  #func= lambda x: 1.0 if abs(x[0,0]**2+x[1,0]**2)<2.0 else 0.0

  x0= np.mat([1.0,1.0]).T
  #w,c= QuadForm(func, x0)
  #w,c= QuadForm2(func, x0)
  w,c= QuadForm3(func, x0)
  A= np.mat([[w[0,0],w[1,0]],[w[1,0],w[2,0]]])
  b= np.mat([[w[3,0]],[w[4,0]]])
  print 'A=',A
  print 'b=',b
  print 'c=',c

  fp= file('/tmp/true.dat','w')
  for x1 in FRange1(-3.0,3.0,50):
    for x2 in FRange1(-3.0,3.0,50):
      y= func(np.mat([x1,x2]).T)
      fp.write('%f %f %f\n' % (x1,x2,y))
    fp.write('\n')
  fp.close()

  fp= file('/tmp/approx.dat','w')
  for x1 in FRange1(-3.0,3.0,50):
    for x2 in FRange1(-3.0,3.0,50):
      x= np.mat([x1,x2]).T
      y= ( np.mat([c]) + b.T*(x-x0) + 0.5*(x-x0).T*A*(x-x0) )[0,0]
      fp.write('%f %f %f\n' % (x1,x2,y))
    fp.write('\n')
  fp.close()

  print 'Plot with'
  print '''qplot -x -3d /tmp/true.dat w l /tmp/approx.dat w l'''



def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa -3d /tmp/true.dat w l /tmp/approx.dat w l &''',
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
