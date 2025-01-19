#!/usr/bin/python3
#\file    ellipse_fit8.py
#\brief   Simplified fitting ellipse method with SVD.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.05, 2023
import numpy as np
import scipy.optimize

def EllipseFit2D(XY):
  XY= np.array(XY)
  centroid= np.average(XY,axis=0)
  U,S,V= np.linalg.svd((np.array(XY)-centroid).T)
  r1,r2= np.sqrt(2.0/len(XY))*S
  angle= np.arctan2(U[0,1],U[0,0])
  cx,cy= centroid
  return [cx,cy],r1,r2,angle

if __name__=='__main__':
  wrand= 0.2
  c= [-9.99,2.3]
  r1,r2= 0.8,0.5
  angle= np.pi/3.0
  print('ground-truth:',c,r1,r2,angle)
  XY=[]
  with open('/tmp/data.dat','w') as fp:
    #for th in np.linspace(0.0,2.0*np.pi,50):
    for th in np.linspace(0.0,2.0*np.pi,10):
    #for th in np.linspace(0.6*np.pi,0.9*np.pi,100):
    #for th in np.linspace(0.6*np.pi,0.9*np.pi,10):
    #for th in np.linspace(0.6*np.pi,0.9*np.pi,5):
      x= c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th) + np.random.uniform(-wrand,wrand)
      y= c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th) + np.random.uniform(-wrand,wrand)
      XY.append([x,y])
      fp.write('%f %f\n'%(x,y))

  c,r1,r2,angle= EllipseFit2D(XY)
  print('estimated:',c,r1,r2,angle)
  with open('/tmp/fit.dat','w') as fp:
    for th in np.linspace(0, 2*np.pi, 1000):
      x= c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th)
      y= c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th)
      fp.write('%f %f\n'%(x,y))

  print('#Plot by:')
  print('''qplot -x /tmp/data.dat /tmp/fit.dat w l''')
