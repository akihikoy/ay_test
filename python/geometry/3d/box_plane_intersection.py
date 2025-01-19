#!/usr/bin/python3
#\file    box_plane_intersection.py
#\brief   Get an intersection polygon between box and plane.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.02, 2021
import numpy as np
from scipy.spatial import ConvexHull as scipy_ConvexHull
from geometry import *

def BoxPlaneIntersection(box, x_box, x_plane):
  EPS= 1.0e-100
  W,D,H= box
  box_points= [[-W*0.5, -D*0.5, -H*0.5],
               [ W*0.5, -D*0.5, -H*0.5 ],
               [ W*0.5,  D*0.5, -H*0.5],
               [-W*0.5,  D*0.5, -H*0.5],
               [-W*0.5, -D*0.5,  H*0.5],
               [ W*0.5, -D*0.5,  H*0.5 ],
               [ W*0.5,  D*0.5,  H*0.5],
               [-W*0.5,  D*0.5,  H*0.5]]
  #Project box_points onto the x_plane frame:
  l_box_points= [TransformLeftInv(x_plane,Transform(x_box,p)) for p in box_points]

  #Indexes of box edges.
  box_edges= [[0,1],[1,2],[2,3],[3,0],
              [4,5],[5,6],[6,7],[7,4],
              [1,5],[4,0],[3,7],[6,2]]
  #Extract box edges that have an intersection with the plane.
  box_edges= [i1_i2 for i1_i2 in box_edges if l_box_points[i1_i2[0]][2]<=0<=l_box_points[i1_i2[1]][2] or l_box_points[i1_i2[1]][2]<=0<=l_box_points[i1_i2[0]][2]]
  if len(box_edges)==0:  return []
  #Calculate intersection points.
  f_intersect= lambda p1,p2: [(p1[0]*p2[2]-p1[2]*p2[0])/(p2[2]-p1[2]), (p1[1]*p2[2]-p1[2]*p2[1])/(p2[2]-p1[2])] if abs(p2[2]-p1[2])>EPS else [(p1[0]+p2[0])*0.5, (p1[1]+p2[1])*0.5]
  l_p_intersect= [f_intersect(l_box_points[i1_i21[0]],l_box_points[i1_i21[1]]) for i1_i21 in box_edges]

  #Make it convex:
  hull= scipy_ConvexHull(l_p_intersect)
  #print hull.vertices
  l_p_intersect= np.array(l_p_intersect)[hull.vertices]

  return l_p_intersect


if __name__=='__main__':
  import matplotlib.pyplot as pyplot
  from plot_cube2 import PlotCube
  from plot_plane import PlotPlane, PlotPoly
  W,D,H= np.random.uniform(0,2,3)
  x_box= np.random.uniform(-1,1,3).tolist() + QFromAxisAngle(np.random.uniform(0,1,3),np.random.uniform(-np.pi,np.pi)).tolist()

  x_plane= np.random.uniform(-1,1,3).tolist() + QFromAxisAngle(np.random.uniform(0,1,3),np.random.uniform(-np.pi,np.pi)).tolist()

  #W,D,H= [0.17, 0.21, 0.13]
  #x_box= [0.5089655172413793, 0.14206896551724135, -0.085, -0.029221169523695276, 0.7065027411494363, 0.029221169523695276, 0.7065027411494363]
  #x_plane= [0.49, 0.03, -0.17, 0.0, 0.0, 0.0, 1.0]

  l_p_intersect= BoxPlaneIntersection([W,D,H], x_box, x_plane)
  print('l_p_intersect:',l_p_intersect)
  p_intersect= [Transform(x_plane,list(l_p)+[0]) for l_p in l_p_intersect]

  fig= pyplot.figure()
  ax= fig.add_subplot(111, projection='3d')
  PlotCube(ax, [W,D,H], x_box)
  PlotPlane(ax, x_plane)
  if len(p_intersect)>0:
    PlotPoly(ax, p_intersect)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim([-1.5,1.5])
  ax.set_ylim([-1.5,1.5])
  ax.set_zlim([-1.5,1.5])
  pyplot.show()
