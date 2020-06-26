#!/usr/bin/python
#\file    ellipse_fit3.py
#\brief   Fitting 2d points with ellipse.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.26, 2020
import numpy as np

#src. http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

def EllipseFit2D(XY):
  centroid= np.average(XY,0) # the centroid of the data set
  x= np.array([[XY[d][0]-centroid[0]] for d in range(len(XY))]) #centering data
  y= np.array([[XY[d][1]-centroid[1]] for d in range(len(XY))]) #centering data
  D= np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
  scale= np.max(np.abs(D),axis=0)
  print 'scale',scale
  D= D/scale
  S= np.dot(D.T,D)
  C= np.zeros([6,6])
  C[0,2]= C[2,0]= 2; C[1,1]= -1
  reg= 0.01
  E,V= np.linalg.eig(np.dot(np.linalg.inv(S+np.eye(S.shape[0])*reg), C))
  print 'np.linalg.inv(S)',np.linalg.inv(S+np.eye(S.shape[0])*reg)
  n= np.argmax(np.abs(E))
  a= V[:,n]
  if a[0]<0:  a= -a
  a= a/scale
  print 'a',a
  #U,S,V= np.linalg.svd(np.dot(np.linalg.inv(S+np.eye(S.shape[0])*reg), C))
  #a= U[:,0]
  #if a[0]<0:  a= -a
  #a= a/scale
  #print 'a',a

  def ellipse_center(a,centroid):
    b,c,d,f,g,a= a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num= b*b-a*c
    x0= (c*d-b*f)/num
    y0= (a*f-b*d)/num
    return [x0+centroid[0],y0+centroid[1]]

  def ellipse_angle_of_rotation(a):
    b,c,d,f,g,a= a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
      if a > c:
        return 0
      else:
        return np.pi/2
    else:
      if a > c:
        return np.arctan(2*b/(a-c))/2
      else:
        return np.pi/2 + np.arctan(2*b/(a-c))/2

  def ellipse_axis_length( a ):
    b,c,d,f,g,a= a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up= 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1= (b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2= (b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    print 'debug',up,down1,down2
    r1= np.sqrt(up/down1)
    r2= np.sqrt(up/down2)
    return r1, r2

  c= ellipse_center(a,centroid)
  r1,r2= ellipse_axis_length(a)
  angle= ellipse_angle_of_rotation(a)
  return c,r1,r2,angle


if __name__=='__main__':
  wrand= 0.002
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


