#!/usr/bin/python
#\file    plot_plane.py
#\brief   Plot 3D plane.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.03, 2021
import numpy as np
import scipy.spatial
from geometry import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def Generate2DPoly(min,max,N=20):
  points_2d= np.random.uniform(min,max,[N,2])
  #Make points_2d convex:
  hull= scipy.spatial.ConvexHull(points_2d)
  return np.array(points_2d)[hull.vertices]

def PlotPlane(ax, x_plane, w=2.0, col='blue'):
  c= np.array(x_plane[:3])
  ex,ey,ez= RotToExyz(QToRot(x_plane[3:]))
  ex,ey= w*ex,w*ey
  points= [c+ex+ey, c-ex+ey, c-ex-ey, c+ex-ey]
  verts= [[points[0],points[1],points[2],points[3]]]
  ax.add_collection3d(Poly3DCollection(verts, facecolors=col, linewidths=0, alpha=.25))

def PlotPoly(ax, points_3d, col=['green','r']):
  ax.add_collection3d(Poly3DCollection([points_3d], facecolors=col[0], linewidths=2, edgecolors=col[1], alpha=.25))

if __name__=='__main__':
  import matplotlib.pyplot as pyplot
  x_plane= np.random.uniform(-1,1,3).tolist() + QFromAxisAngle(np.random.uniform(0,1,3),np.random.uniform(-np.pi,np.pi)).tolist()

  l_points2d_poly= Generate2DPoly([-1,-1],[1,1])
  points2d_poly= map(lambda l_p: Transform(x_plane,list(l_p)+[0]), l_points2d_poly)

  fig= pyplot.figure()
  ax= fig.add_subplot(111, projection='3d')
  PlotPlane(ax, x_plane)
  PlotPoly(ax, points2d_poly)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim([-1.5,1.5])
  ax.set_ylim([-1.5,1.5])
  ax.set_zlim([-1.5,1.5])
  pyplot.show()
