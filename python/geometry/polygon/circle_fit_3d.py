#!/usr/bin/python
#3D-circle fitting algirithm
import numpy as np
import numpy.linalg as la
from circle_fit import CircleFit2D
from pca2 import TPCA, TPCA_SVD

#Fitting a circle to the data XYZ, return the center [x,y,z], the radius, and the rotation matrix.
#Note: ez[2] of the rotation matrix will be always positive.
def CircleFit3D(XYZ):
  pca= TPCA(XYZ)
  XY= pca.Projected[:,[0,1]]
  c,r= CircleFit2D(XY)
  ex= pca.EVecs[:,0]
  ey= pca.EVecs[:,1]
  ez= np.cross(ex,ey)
  if ez[2]<0.0:
    ex=-ex; ey=-ey; ez=-ez
  Rot= np.zeros((3,3))
  Rot[:,0],Rot[:,1],Rot[:,2]= ex,ey,ez
  return pca.Reconstruct(c,[0,1]), r, Rot


import sys,os
filedir= os.path.dirname(os.path.abspath(__file__))
sys.path.append(filedir+'/lfd_trick')
from base.base_geom import *

def Main():
  from random import random, uniform
  wrand= 0.05
  c= [-9.99,2.3,0.4]
  r= 0.5
  #Rot= QToRot(QFromAxisAngle([1.0,1.0,0.0],DegToRad(30.0)))
  Rot= QToRot(QFromAxisAngle([1.0,1.0,0.0],uniform(-math.pi,math.pi)))
  print 'ground-truth:',c,r,'\n',Rot
  print '|ex|,|ey|,|ez|:',Norm(Rot[:,0]),Norm(Rot[:,1]),Norm(Rot[:,2])
  XYZ=[]
  fp= open('/tmp/data.dat','w')
  #for th in FRange1(0.6*math.pi,0.9*math.pi,100):
  for th in FRange1(0.2*math.pi,0.9*math.pi,10):
  #for th in FRange1(0.0,2.0*math.pi,50):
    x= r*math.cos(th)+(random()-0.5)*wrand
    y= r*math.sin(th)+(random()-0.5)*wrand
    z= (random()-0.5)*wrand
    xyz= np.array(c)+np.dot(Rot,[x,y,z])
    XYZ.append(ToList(xyz))
    fp.write('%f %f %f\n'%(xyz[0],xyz[1],xyz[2]))
  fp.close()

  c,r,Rot= CircleFit3D(XYZ)
  print 'CircleFit:',c,r,'\n',Rot
  print '|ex|,|ey|,|ez|:',Norm(Rot[:,0]),Norm(Rot[:,1]),Norm(Rot[:,2])
  print 'cross(ex,ey),ez:',np.cross(Rot[:,0],Rot[:,1]),Rot[:,2]
  fp= open('/tmp/fit.dat','w')
  for th in FRange1(-math.pi,+math.pi,100):
    x= r*math.cos(th)
    y= r*math.sin(th)
    xyz= Vec(c)+np.dot(Rot,[x,y,0.0])
    fp.write('%f %f %f\n'%(xyz[0],xyz[1],xyz[2]))
  fp.close()

def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa -3d -s 'set size square;set size ratio -1'
          /tmp/fit.dat w l
          /tmp/data.dat w p
          &''',
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
