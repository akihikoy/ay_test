#!/usr/bin/python
#\file    ellipse_fit1.py
#\brief   Fitting 2d points with ellipse.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.26, 2020
import numpy as np

#src. https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points

def EllipseFit2D(XY):
  centroid= np.average(XY,0) # the centroid of the data set
  X= [XY[d][0]-centroid[0] for d in range(len(XY))] # centering data
  Y= [XY[d][1]-centroid[1] for d in range(len(XY))] # centering data
  U,S,V= np.linalg.svd((np.array(XY)-centroid).T)
  #print "U=",U
  #print "S=",S
  #print "V=",V

  r1,r2= np.sqrt(2.0/len(XY)) * S
  angle= np.arctan2(U[0,1],U[0,0])
  return centroid,r1,r2,angle


if __name__=='__main__':
  wrand= 0.01
  c= [-9.99,2.3]
  r1,r2= 0.8,0.5
  angle= np.pi/3.0
  print 'ground-truth:',c,r1,r2,angle
  XY=[]
  with open('/tmp/data.dat','w') as fp:
    #for th in np.linspace(0.0,2.0*np.pi,50):
    for th in np.linspace(0.6*np.pi,0.9*np.pi,100):
    #for th in np.linspace(0.6*np.pi,0.9*np.pi,10):
    #for th in np.linspace(0.6*np.pi,0.9*np.pi,3):
      x= c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th) + np.random.uniform(-wrand,wrand)
      y= c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th) + np.random.uniform(-wrand,wrand)
      XY.append([x,y])
      fp.write('%f %f\n'%(x,y))

  #U,S,centroid= EllipseFit2D(XY)
  #with open('/tmp/fit.dat','w') as fp:
    #tt= np.linspace(0, 2*np.pi, 1000)
    #circle= np.stack((np.cos(tt), np.sin(tt)))    # unit circle
    #transform= np.sqrt(2.0/len(XY)) * U.dot(np.diag(S))   # transformation matrix
    #fit= transform.dot(circle).T + centroid
    #for x,y in fit:
      #fp.write('%f %f\n'%(x,y))

  c,r1,r2,angle= EllipseFit2D(XY)
  print 'estimated:',c,r1,r2,angle
  with open('/tmp/fit.dat','w') as fp:
    for th in np.linspace(0, 2*np.pi, 1000):
      x= c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th)
      y= c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th)
      fp.write('%f %f\n'%(x,y))


