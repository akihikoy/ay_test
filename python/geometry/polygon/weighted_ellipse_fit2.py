#!/usr/bin/python3
#\file    weighted_ellipse_fit2.py
#\brief   Sample-weighted ellipse fitting.
#         Based on ellipse_fit1:
#         Geometric fitting with SVD.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.29, 2020
import numpy as np
import scipy.optimize
from weighted_ellipse_fit1 import SqErrorFromEllipse

def SampleWeightedEllipseFit2D(XY, W, centroid=None):
  sumw= np.sum(W)
  maxw= np.max(W)
  if centroid is None:
    centroid= np.average(np.diag(W).dot(XY),0)*(len(XY)/sumw)
  U,S,V= np.linalg.svd(np.diag(W/maxw).dot((np.array(XY)-centroid)).T)
  #print "U=",U
  #print "S=",S
  #print "V=",V

  r1,r2= np.sqrt(2.0*maxw/sumw) * S
  angle= np.arctan2(U[0,1],U[0,0])
  return centroid,r1,r2,angle


if __name__=='__main__':
  wrand= 0.05
  c1= [0.1,0.15]
  r11,r12= 0.2,0.15
  angle1= np.pi/6.0
  c2= [0.55,0.2]
  r21,r22= 0.25,0.2
  angle2= -np.pi/8.0
  print('ground-truth(1):',c1,r11,r12,angle1)
  print('ground-truth(2):',c2,r21,r22,angle2)
  XY=[]
  W=[]
  w_scale= 100.0
  with open('/tmp/data.dat','w') as fp:
    c,r1,r2,angle= c1,r11,r12,angle1
    for th in np.linspace(np.pi/6.0,(2.0-1.0/6.0)*np.pi,50):
      x= c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th) + np.random.uniform(-wrand,wrand)
      y= c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th) + np.random.uniform(-wrand,wrand)
      XY.append([x,y])
      fp.write('%f %f\n'%(x,y))
      W.append(1.0/(1.0+w_scale*SqErrorFromEllipse([x,y],c1,r11,r12,angle1)))
    fp.write('\n')
    c,r1,r2,angle= c2,r21,r22,angle2
    for th in np.linspace((-1.0+0.25)*np.pi,1.0*np.pi,50):
      x= c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th) + np.random.uniform(-wrand,wrand)
      y= c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th) + np.random.uniform(-wrand,wrand)
      XY.append([x,y])
      fp.write('%f %f\n'%(x,y))
      W.append(1.0/(1.0+w_scale*SqErrorFromEllipse([x,y],c1,r11,r12,angle1)))
  print('W',W)

  #from ellipse_fit1 import EllipseFit2D
  #from ellipse_fit2 import EllipseFit2D
  #from ellipse_fit3 import EllipseFit2D
  #from ellipse_fit4 import EllipseFit2D
  #from ellipse_fit5 import EllipseFit2D
  #from ellipse_fit6 import EllipseFit2D
  #from ellipse_fit7 import EllipseFit2D
  #c,r1,r2,angle= EllipseFit2D(XY)
  c,r1,r2,angle= SampleWeightedEllipseFit2D(XY, W)  #centroid=np.array([0.,0.])
  print('estimated:',c,r1,r2,angle)
  with open('/tmp/fit.dat','w') as fp:
    for th in np.linspace(0, 2*np.pi, 1000):
      x= c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th)
      y= c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th)
      fp.write('%f %f\n'%(x,y))

  print('#Plot by:')
  print('''qplot -x /tmp/data.dat /tmp/fit.dat w l''')
