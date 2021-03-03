#!/usr/bin/python
#\file    box_line_intersection.py
#\brief   Get an intersection part of a 3D oriented box and a line segment.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.03, 2021
import numpy as np
from geometry import *
from box_ray_intersection import BoxRayIntersection

def BoxLineIntersection(box, x_box, p1, p2):
  W,D,H= box
  l_p1= TransformLeftInv(x_box,p1)
  l_p2= TransformLeftInv(x_box,p2)
  ray_o= np.array(l_p1)
  ray_d= np.array(l_p2)-ray_o
  line_len= np.linalg.norm(ray_d)
  ray_d/= line_len
  box_min= [-W*0.5, -D*0.5, -H*0.5]
  box_max= [ W*0.5,  D*0.5,  H*0.5]
  tmin,tmax= BoxRayIntersection(ray_o, ray_d, box_min, box_max)
  #print 'p1,p2:',p1,p2
  #print 'ray_o:',ray_o
  #print 'ray_d:',ray_d
  #print 'box_min:',box_min
  #print 'box_max:',box_max
  #print 'tmin,tmax:',tmin,tmax
  if tmin is None:  return None,None
  if tmax<0 or tmin>line_len:  return None,None
  if tmin<0:  tmin= 0.0
  if tmax>line_len:  tmax= line_len
  return Transform(x_box,ray_o+tmin*ray_d),Transform(x_box,ray_o+tmax*ray_d)

if __name__=='__main__':
  import matplotlib.pyplot as pyplot
  from plot_cube2 import PlotCube
  from plot_line import PlotLine
  W,D,H= np.random.uniform(0,2,3)
  x_box= np.random.uniform(-1,1,3).tolist() + QFromAxisAngle(np.random.uniform(0,1,3),np.random.uniform(-np.pi,np.pi)).tolist()

  p1= np.random.uniform(-1,1,3)
  p2= np.random.uniform(-1,1,3)

  pi1,pi2= BoxLineIntersection([W,D,H], x_box, p1, p2)
  print 'Intersection line segment:',pi1,pi2

  fig= pyplot.figure()
  ax= fig.add_subplot(111, projection='3d')
  PlotCube(ax, [W,D,H], x_box)
  PlotLine(ax, p1, p2, col='blue')
  if pi1 is not None:
    PlotLine(ax, pi1, pi2, lw=3, col='red')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim([-1.5,1.5])
  ax.set_ylim([-1.5,1.5])
  ax.set_zlim([-1.5,1.5])
  pyplot.show()
