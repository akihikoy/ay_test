#!/usr/bin/python
#\file    ellipse_fit7.py
#\brief   Fitting 2d points with ellipse.
#         Geometric fitting with optimization.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.28, 2020
import numpy as np
import scipy.optimize

def EllipseFit2D(XY):
  centroid= np.average(XY,0) # the centroid of the data set
  X= [XY[d][0]-centroid[0] for d in range(len(XY))] # centering data
  Y= [XY[d][1]-centroid[1] for d in range(len(XY))] # centering data
  U,S,V= np.linalg.svd((np.array(XY)-centroid).T)
  r1,r2= np.sqrt(2.0/len(XY)) * S
  angle= np.arctan2(U[0,1],U[0,0])
  x0= [centroid[0],centroid[1],r1,r2,angle]

  def calc_error(x):
    cx,cy,r1,r2,angle= x
    err= 0.0
    ca= np.cos(angle)
    sa= np.sin(angle)
    for x,y in XY:
      x0= ca*(x-cx)+sa*(y-cy)
      y0= -sa*(x-cx)+ca*(y-cy)
      th= np.arctan2(y0/r2,x0/r1)
      x= r1*np.cos(th)
      y= r2*np.sin(th)
      err+= (x-x0)**2 + (y-y0)**2
    #print 'err',err
    return err

  bounds= ((None,None),(None,None),(0.0,None),(0.0,None),(-np.pi,np.pi))
  print 'x0',x0,calc_error(x0)
  print 'bounds',bounds
  res= scipy.optimize.minimize(calc_error,x0,bounds=bounds)
  cx,cy,r1,r2,angle= res.x
  return [cx,cy],r1,r2,angle


if __name__=='__main__':
  wrand= 0.05
  c= [-9.99,2.3]
  r1,r2= 0.8,0.5
  angle= np.pi/3.0
  print 'ground-truth:',c,r1,r2,angle
  XY=[]
  with open('/tmp/data.dat','w') as fp:
    #for th in np.linspace(0.0,2.0*np.pi,50):
    #for th in np.linspace(0.6*np.pi,0.9*np.pi,100):
    #for th in np.linspace(0.6*np.pi,0.9*np.pi,10):
    for th in np.linspace(0.6*np.pi,0.9*np.pi,5):
      x= c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th) + np.random.uniform(-wrand,wrand)
      y= c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th) + np.random.uniform(-wrand,wrand)
      XY.append([x,y])
      fp.write('%f %f\n'%(x,y))

  c,r1,r2,angle= EllipseFit2D(XY)
  print 'estimated:',c,r1,r2,angle
  with open('/tmp/fit.dat','w') as fp:
    for th in np.linspace(0, 2*np.pi, 1000):
      x= c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th)
      y= c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th)
      fp.write('%f %f\n'%(x,y))


