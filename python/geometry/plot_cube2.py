#!/usr/bin/python
#\file    plot_cube2.py
#\brief   Plot oriented cube test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.02, 2021
import numpy as np
from geometry import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as pyplot
from plot_cube import PlotCube

if __name__=='__main__':
  W,D,H= np.random.uniform(0,2,3)
  cube_points= [[-W*0.5, -D*0.5, -H*0.5],
                [ W*0.5, -D*0.5, -H*0.5 ],
                [ W*0.5,  D*0.5, -H*0.5],
                [-W*0.5,  D*0.5, -H*0.5],
                [-W*0.5, -D*0.5,  H*0.5],
                [ W*0.5, -D*0.5,  H*0.5 ],
                [ W*0.5,  D*0.5,  H*0.5],
                [-W*0.5,  D*0.5,  H*0.5]]

  x_center= np.random.uniform(-1,1,3).tolist() + QFromAxisAngle(np.random.uniform(0,1,3),np.random.uniform(-np.pi,np.pi)).tolist()
  cube_points= np.array(map(lambda p: Transform(x_center,p), cube_points))

  fig= pyplot.figure()
  ax= fig.add_subplot(111, projection='3d')
  PlotCube(ax, cube_points)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim([-1.5,1.5])
  ax.set_ylim([-1.5,1.5])
  ax.set_zlim([-1.5,1.5])
  pyplot.show()
