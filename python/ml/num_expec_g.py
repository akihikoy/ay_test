#!/usr/bin/python
#\file    num_expec_g.py
#\brief   Sampling based numerical expectation.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.17, 2015
import math
import numpy as np

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Gaussian(x, x0, var):
  diff= x - x0
  #return math.exp(-diff*diff/(2.0*var)) / math.sqrt(2*math.pi*var)
  return math.exp(-diff*diff/(2.0*var))

#Get expectation of f numerically w.r.t. Gaussian(x0,var)
def NumExpec1(f, x0, var, nsd=4.0, N=200):
  sd= math.sqrt(var)
  integ= 0.0
  sumG= 0.0
  for x in FRange1(x0-nsd*sd, x0+nsd*sd, N):
    g= Gaussian(x, x0, var)
    integ+= f(x)*g
    sumG+= g
  return integ/sumG

#Get expectation of f numerically w.r.t. Gaussian(x0,var)
def NumExpec2(f, x0, var, N=200):
  sd= math.sqrt(var)
  integ= 0.0
  for i in range(N):
    x= np.random.multivariate_normal([x0], [[var]])
    integ+= f(x)  #*Gaussian(x, x0, var)*dx
  return integ/float(N)


def Main():
  #TrueFunc= lambda x: 0.5*x
  #TrueFunc= lambda x: 1.2+math.sin(x)
  #TrueFunc= lambda x: 1.2+math.sin(3*x)
  #TrueFunc= lambda x: 2.0*x**2
  #TrueFunc= lambda x: 4.0-x if x>0.0 else 0.0
  TrueFunc= lambda x: 4.0 if 0.0<x and x<2.5 else 0.0

  Bound= [-3.0,5.0]

  x_test= np.array([[x] for x in FRange1(*Bound,num_div=100)]).astype(np.float32)
  y_test= np.array([[TrueFunc(x[0])] for x in x_test]).astype(np.float32)
  var= 0.5**2
  y_exp_test= np.array([[NumExpec2(TrueFunc,x[0],var)] for x in x_test]).astype(np.float32)

  fp1= file('/tmp/exp_true.dat','w')
  for x,y in zip(x_test,y_test):
    fp1.write('%s %s\n' % (' '.join(map(str,x)),' '.join(map(str,y))))
  fp1.close()

  fp1= file('/tmp/exp_05.dat','w')
  for x,y in zip(x_test,y_exp_test):
    fp1.write('%s %s\n' % (' '.join(map(str,x)),' '.join(map(str,y))))
  fp1.close()


def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa /tmp/exp_true.dat w l /tmp/exp_05.dat w l &''',
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
