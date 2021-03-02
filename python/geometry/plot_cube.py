#!/usr/bin/python
#\file    plot_cube.py
#\brief   Plot cube test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.09, 2019
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as pyplot

def PlotCube(ax, cube_points):
  #plot vertices
  ax.scatter3D(cube_points[:, 0], cube_points[:, 1], cube_points[:, 2])
  #list of sides' polygons of figure
  verts= [[cube_points[0],cube_points[1],cube_points[2],cube_points[3]],
          [cube_points[4],cube_points[5],cube_points[6],cube_points[7]], 
          [cube_points[0],cube_points[1],cube_points[5],cube_points[4]], 
          [cube_points[2],cube_points[3],cube_points[7],cube_points[6]], 
          [cube_points[1],cube_points[2],cube_points[6],cube_points[5]],
          [cube_points[4],cube_points[7],cube_points[3],cube_points[0]]]
  #plot sides
  ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))


if __name__=='__main__':
  cube_points= np.array([[-1, -1, -1],
                        [1, -1, -1 ],
                        [1, 1, -1],
                        [-1, 1, -1],
                        [-1, -1, 1],
                        [1, -1, 1 ],
                        [1, 1, 1],
                        [-1, 1, 1]])
  fig= pyplot.figure()
  ax= fig.add_subplot(111, projection='3d')
  PlotCube(ax, cube_points)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  pyplot.show()

