#!/usr/bin/python3
#\file    svd_geom.py
#\brief   SVD geometry visualization.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.12, 2020
import numpy as np
from polygon.ellipse_fit1 import EllipseFit2D

#def EllipseFit2D(XY):
  #centroid= np.average(XY,0) # the centroid of the data set
  #U,S,V= np.linalg.svd((np.array(XY)-centroid).T)
  ##print "U=",U
  ##print "S=",S
  ##print "V=",V

  #r1,r2= np.sqrt(2.0/len(XY)) * S
  #angle= np.arctan2(U[0,1],U[0,0])
  #return centroid,r1,r2,angle


if __name__=='__main__':
  wrand= 0.1
  c= [-9.99,2.3]
  r1,r2= 0.8,0.5
  angle= np.pi/3.0
  print('ground-truth:',c,r1,r2,angle)
  XY=[]
  with open('/tmp/data.dat','w') as fp:
    for th in np.linspace(0.0,2.0*np.pi,50):
    #for th in np.linspace(0.6*np.pi,0.9*np.pi,100):
    #for th in np.linspace(0.6*np.pi,0.9*np.pi,10):
    #for th in np.linspace(0.6*np.pi,0.9*np.pi,3):
      x= c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th) + np.random.uniform(-wrand,wrand)
      y= c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th) + np.random.uniform(-wrand,wrand)
      XY.append([x,y])
      fp.write('%f %f\n'%(x,y))

  centroid= np.average(XY,0) # the centroid of the data set
  U,S,V= np.linalg.svd((np.array(XY)-centroid).T)
  S*= np.sqrt(2.0/len(XY))
  print('U=',U)
  print('S=',S)
  print('V=',V)

  with open('/tmp/svd.dat','w') as fp:
    write_vec2= lambda v: fp.write('%f %f\n'%(v[0],v[1]))
    write_vec2(centroid)
    write_vec2(centroid+S[0]*U[:,0])
    fp.write('\n\n')
    write_vec2(centroid)
    write_vec2(centroid+S[1]*U[:,1])

  c,r1,r2,angle= EllipseFit2D(XY)
  print('estimated:',c,r1,r2,angle)
  with open('/tmp/ellipse.dat','w') as fp:
    for th in np.linspace(0, 2*np.pi, 1000):
      x= c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th)
      y= c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th)
      fp.write('%f %f\n'%(x,y))

  print('#Plot by:')
  print('''qplot -x -s 'set size ratio -1' /tmp/data.dat /tmp/ellipse.dat w l /tmp/svd.dat w l''')
