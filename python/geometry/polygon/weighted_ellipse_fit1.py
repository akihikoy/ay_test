#!/usr/bin/python
#\file    weighted_ellipse_fit1.py
#\brief   Sample-weighted ellipse fitting.
#         Based on ellipse_fit5:
#         Algebraic ellipse fitting using linear least squares with bounds.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.29, 2020
import numpy as np
import scipy.optimize


def SqErrorFromEllipse(x, c,r1,r2,angle):
  ca= np.cos(angle)
  sa= np.sin(angle)
  x0= ca*(x[0]-c[0])+sa*(x[1]-c[1])
  y0= -sa*(x[0]-c[0])+ca*(x[1]-c[1])
  th= np.arctan2(y0/r2,x0/r1)
  return (r1*np.cos(th)-x0)**2 + (r2*np.sin(th)-y0)**2

#Based on ellipse_fit5:
#Algebraic ellipse fitting using linear least squares with bounds.
def SampleWeightedEllipseFit2D(XY, W):
  centroid= np.average(XY,0) # the centroid of the data set
  x= np.array([[XY[d][0]-centroid[0]] for d in range(len(XY))]) #centering data
  y= np.array([[XY[d][1]-centroid[1]] for d in range(len(XY))]) #centering data
  A= np.dot(np.diag(W), np.hstack([x*x, x*y, y*y, x, y]))
  b= W*np.ones(x.shape[0])
  #x= np.linalg.lstsq(A, b)[0].squeeze()
  bounds= ([0.0001,-np.inf,0.0001,-np.inf,-np.inf], np.inf)
  x= scipy.optimize.lsq_linear(A, b, bounds=bounds).x
  a= np.hstack((x,[-1.0]))
  print 'a',a
  if a[1]**2-4.*a[0]*a[2]>0.0:
    print '####################################'
    print 'Warning: Invalid ellipse parameters.'
    print '####################################'
    return None

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
  wrand= 0.05
  c1= [0.1,0.15]
  r11,r12= 0.2,0.15
  angle1= np.pi/6.0
  c2= [0.55,0.2]
  r21,r22= 0.25,0.2
  angle2= -np.pi/8.0
  print 'ground-truth(1):',c1,r11,r12,angle1
  print 'ground-truth(2):',c2,r21,r22,angle2
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
  print 'W',W

  #from ellipse_fit1 import EllipseFit2D
  #from ellipse_fit2 import EllipseFit2D
  #from ellipse_fit3 import EllipseFit2D
  #from ellipse_fit4 import EllipseFit2D
  #from ellipse_fit5 import EllipseFit2D
  #from ellipse_fit6 import EllipseFit2D
  #from ellipse_fit7 import EllipseFit2D
  #c,r1,r2,angle= EllipseFit2D(XY)
  W= np.ones_like(W); print 'modifying W',W
  c,r1,r2,angle= SampleWeightedEllipseFit2D(XY, W)  #If we don't use W= np.ones_like(W), it does not converges.
  print 'estimated:',c,r1,r2,angle
  with open('/tmp/fit.dat','w') as fp:
    for th in np.linspace(0, 2*np.pi, 1000):
      x= c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th)
      y= c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th)
      fp.write('%f %f\n'%(x,y))

  print '#Plot by:'
  print '''qplot -x /tmp/data.dat /tmp/fit.dat w l'''
