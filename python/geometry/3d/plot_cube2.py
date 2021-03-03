#!/usr/bin/python
#\file    plot_cube2.py
#\brief   Plot oriented cube.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.02, 2021
import numpy as np
from geometry import *
import plot_cube

def PlotCube(ax, cube, x_cube):
  W,D,H= cube
  cube_points= [[-W*0.5, -D*0.5, -H*0.5],
                [ W*0.5, -D*0.5, -H*0.5 ],
                [ W*0.5,  D*0.5, -H*0.5],
                [-W*0.5,  D*0.5, -H*0.5],
                [-W*0.5, -D*0.5,  H*0.5],
                [ W*0.5, -D*0.5,  H*0.5 ],
                [ W*0.5,  D*0.5,  H*0.5],
                [-W*0.5,  D*0.5,  H*0.5]]
  cube_points= np.array(map(lambda p: Transform(x_cube,p), cube_points))
  plot_cube.PlotCube(ax, cube_points)


if __name__=='__main__':
  import matplotlib.pyplot as pyplot
  W,D,H= np.random.uniform(0,2,3)
  x_center= np.random.uniform(-1,1,3).tolist() + QFromAxisAngle(np.random.uniform(0,1,3),np.random.uniform(-np.pi,np.pi)).tolist()

  fig= pyplot.figure()
  ax= fig.add_subplot(111, projection='3d')
  PlotCube(ax, [W,D,H], x_center)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim([-1.5,1.5])
  ax.set_ylim([-1.5,1.5])
  ax.set_zlim([-1.5,1.5])
  pyplot.show()
