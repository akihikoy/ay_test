#!/usr/bin/python
#\file    box_poly_intersection.py
#\brief   Get an intersection polygon between a box and a polygon on a plane.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.03, 2021
import numpy as np
import scipy.spatial
from geometry import *
from box_plane_intersection import BoxPlaneIntersection
from polygon_clip import ClipPolygon

def BoxPolyIntersection(box, x_box, x_poly, l_points2d_poly):
  l_points2d_boxpolyintersect= BoxPlaneIntersection(box, x_box, x_poly)
  l_points2d_intersect= ClipPolygon(l_points2d_boxpolyintersect, l_points2d_poly)
  return l_points2d_intersect

if __name__=='__main__':
  import matplotlib.pyplot as pyplot
  from plot_cube2 import PlotCube
  from plot_plane import PlotPoly, Generate2DPoly
  W,D,H= np.random.uniform(0,2,3)
  x_box= np.random.uniform(-1,1,3).tolist() + QFromAxisAngle(np.random.uniform(0,1,3),np.random.uniform(-np.pi,np.pi)).tolist()

  x_poly= np.random.uniform(-1,1,3).tolist() + QFromAxisAngle(np.random.uniform(0,1,3),np.random.uniform(-np.pi,np.pi)).tolist()
  l_points2d_poly= Generate2DPoly([-1,-1],[1,1])
  points2d_poly= map(lambda l_p: Transform(x_poly,list(l_p)+[0]), l_points2d_poly)

  l_p_intersect= BoxPolyIntersection([W,D,H], x_box, x_poly, l_points2d_poly)
  print 'l_p_intersect:',l_p_intersect
  p_intersect= map(lambda l_p: Transform(x_poly,list(l_p)+[0]), l_p_intersect)

  fig= pyplot.figure()
  ax= fig.add_subplot(111, projection='3d')
  PlotCube(ax, [W,D,H], x_box)
  PlotPoly(ax, points2d_poly, ['blue','blue'])
  if len(p_intersect)>0:
    PlotPoly(ax, p_intersect, ['red','red'])
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim([-1.5,1.5])
  ax.set_ylim([-1.5,1.5])
  ax.set_zlim([-1.5,1.5])
  pyplot.show()
