#!/usr/bin/python
#\file    param_Nd.py
#\brief   Parameterized splines on N-dimensional space.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.29, 2015
import random

#Generate a random number of uniform distribution of specified bound.
def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

#Generate a random vector of uniform distribution; each dim has the same bound.
def RandVec(nd,xmin=-0.5,xmax=0.5):
  return Vec([Rand(xmin,xmax) for d in range(nd)])

#Generate a random vector of uniform distribution; each dim has different bound.
def RandN(xmins,xmaxs):
  assert(len(xmins)==len(xmaxs))
  return [Rand(xmins[d],xmaxs[d]) for d in range(len(xmins))]

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

#Generate sample points: [t,x-1,x-2,...,x-Nd]*Ns
def GenNd_1(x0=[0.0],xf=[1.0],tf=1.0,param=[[0.0,0.0]]):
  Nd= len(x0)
  data= [[0.0]+[None]*Nd,
         [0.333*tf]+[None]*Nd,
         [0.666*tf]+[None]*Nd,
         [tf]+[None]*Nd]
  for d in range(Nd):
    a= (xf[d]-x0[d])/tf
    data[0][1+d]= x0[d]
    data[1][1+d]= x0[d]+a*data[1][0] + param[d][0]
    data[2][1+d]= x0[d]+a*data[2][0] + param[d][1]
    data[3][1+d]= xf[d]
  return data

Nd= 3
def Main():
  from cubic_hermite_spline import TCubicHermiteSpline

  data= GenNd_1(x0=[0.0]*Nd, xf=[1.0]*Nd, param=[RandN([-1.0,-1.0],[1.0,1.0]) for d in range(Nd)])

  splines= [TCubicHermiteSpline() for d in range(len(data[0])-1)]
  for d in range(len(splines)):
    data_d= [[x[0],x[d+1]] for x in data]
    splines[d].Initialize(data_d, tan_method=splines[d].CARDINAL, c=0.0, m=0.0)

  pf= open('/tmp/spline1.dat','w')
  t= data[0][0]
  while True:
    x= [splines[d].Evaluate(t) for d in range(len(splines))]
    pf.write('%f %s\n' % (t, ' '.join(map(str,x))))
    if t>data[-1][0]:  break
    t+= 0.02
    #t+= 0.001
  print 'Generated:','/tmp/spline1.dat'

  pf= open('/tmp/spline0.dat','w')
  for d in data:
    pf.write('%s\n' % ' '.join(map(str,d)))
  print 'Generated:','/tmp/spline0.dat'


def PlotGraphs():
  print 'Plotting graphs..'
  import os
  pline= ''
  for d in range(Nd):
    pline+= ''' /tmp/spline1.dat u 1:{d} w l lt {lt}
                /tmp/spline0.dat u 1:{d} w p lt {lt} ps 2'''.format(d=d+2,lt=d+1)
  commands=[
    #'''qplot -x2 aaa
        #/tmp/spline1.dat u 1:2 w l
        #/tmp/spline0.dat u 1:2 w p pt 5 ps 2  &''',
    '''qplot -x2 aaa {pline} &'''.format(pline=pline),
    '''''',
    '''''',
    ]
  if Nd==3:  commands[-1]= '''qplot -x2 aaa -3d /tmp/spline1.dat u 2:3:4 w l /tmp/spline0.dat u 2:3:4 w p pt 5 ps 2 &'''
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
