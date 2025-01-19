#!/usr/bin/python3
#\file    ellipse_point_in_out.py
#\brief   Check if a point is in an ellipse or not.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.03, 2020
import numpy as np

#Check if point=[x,y] is in an ellipse.
def PointInEllipse(point, c,r1,r2,angle):
  ca= np.cos(angle)
  sa= np.sin(angle)
  x0= ca*(point[0]-c[0])+sa*(point[1]-c[1])
  y0= -sa*(point[0]-c[0])+ca*(point[1]-c[1])
  return (x0*x0)/(r1*r1)+(y0*y0)/(r2*r2)<1.0

if __name__=='__main__':
  c= [-1.0,2.3]
  r1,r2= 0.8,0.5
  angle= np.pi/3.0
  XY=[]
  with open('/tmp/orig.dat','w') as fp:
    for th in np.linspace(0.0,2.0*np.pi,200):
      x= c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th)
      y= c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th)
      XY.append([x,y])
      fp.write('%f %f\n'%(x,y))

  def write_in_out(fp1,fp2,x,y):
    p= [x,y]
    inout= PointInEllipse(p, c,r1,r2,angle)
    if inout:
      fp1.write(' '.join(map(str,p))+'\n')
    else:
      fp2.write(' '.join(map(str,p))+'\n')
    print(p,inout)
  with open('/tmp/points_in.dat','w') as fp1:
    with open('/tmp/points_out.dat','w') as fp2:
      for x in np.linspace(-2.0,0.0,50):
        for y in np.linspace(1.3,3.3,50):
          write_in_out(fp1,fp2,x,y)

  print('# Plot by')
  print("qplot -x -s 'set size ratio -1' /tmp/orig.dat w l /tmp/points_in.dat /tmp/points_out.dat")
