#!/usr/bin/python3
#2D-circle fitting algirithm
#Src:
#http://people.cas.uab.edu/~mosya/cl/HyperSVD.m
#http://people.cas.uab.edu/~mosya/cl/MATLABcircle.html

import math
import numpy as np
import numpy.linalg as la

def CircleFit2D(XY):
  centroid= np.average(XY,0) # the centroid of the data set
  #print centroid

  X= [XY[d][0]-centroid[0] for d in range(len(XY))] # centering data
  Y= [XY[d][1]-centroid[1] for d in range(len(XY))] # centering data
  Z= [X[d]**2 + Y[d]**2 for d in range(len(XY))]
  ZXY1= np.matrix([Z, X, Y, [1.0]*len(Z)]).transpose()
  #print ZXY1
  U,S,V= la.svd(ZXY1,0)
  #print "U=",U
  #print "S=",S
  #print "V=",V
  #print S[3]/S[0]
  if S[3]/S[0]<1.0e-12:  # singular case
    print("SINGULAR")
    A= (V.transpose())[:,3]
  else:  # regular case
    R= np.average(np.array(ZXY1),0)
    #print "R=",R
    N= np.matrix([[8.0*R[0], 4.0*R[1], 4.0*R[2], 2.0],
                  [4.0*R[1], 1.0, 0.0, 0.0],
                  [4.0*R[2], 0.0, 1.0, 0.0],
                  [2.0,      0.0, 0.0, 0.0]])
    #print "N=",N
    W= V.transpose()*np.diag(S)*V
    #print "W=",W
    #print "W*la.inv(N)*W=",W*la.inv(N)*W
    D,E= la.eig(W*la.inv(N)*W)  # values, vectors
    #print "E=",E
    #print "D=",D
    idx= D.argsort()
    #print "idx=",idx
    Astar= E[:,idx[1]]
    #print "Astar=",Astar
    A= la.solve(W, Astar)
    #print "A=",A

  #print A
  A= np.array(A)[:,0]
  #print A
  #print -A[1:3]/A[0]/2.0
  #print A[1]**2+A[2]**2-4.0*A[0]*A[3]
  return -A[1:3].transpose()/A[0]/2.0+centroid,  \
          math.sqrt(A[1]**2+A[2]**2-4.0*A[0]*A[3])/abs(A[0])/2.0;


#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Main():
  from random import random
  wrand= 0.05
  c= [-9.99,2.3]
  r= 0.5
  #wrand= 0.00
  #c= [0.0,0.0]
  #r= 0.012
  print('ground-truth:',c,r)
  XY=[]
  fp= open('/tmp/data.dat','w')
  #for th in FRange1(0.6*math.pi,0.9*math.pi,100):
  for th in FRange1(0.6*math.pi,0.9*math.pi,10):
  #for th in FRange1(0.0,2.0*math.pi,50):
    x= c[0]+r*math.cos(th)+np.random.uniform(-wrand,wrand)
    y= c[1]+r*math.sin(th)+np.random.uniform(-wrand,wrand)
    XY.append([x,y])
    fp.write('%f %f\n'%(x,y))
  fp.close()

  c,r= CircleFit2D(XY)
  print('CircleFit:',c,r)
  fp= open('/tmp/fit.dat','w')
  for th in FRange1(-math.pi,+math.pi,100):
    x= c[0]+r*math.cos(th)
    y= c[1]+r*math.sin(th)
    fp.write('%f %f\n'%(x,y))
  fp.close()

def PlotGraphs():
  print('Plotting graphs..')
  import os
  commands=[
    '''qplot -x2 aaa -s 'set size square;set size ratio -1' /tmp/fit.dat w l /tmp/data.dat w p &''',
    '''''',
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

