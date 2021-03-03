#!/usr/bin/python
#\file    plot_line.py
#\brief   Plot 3D line segment.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.03, 2021
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def PlotLine(ax, p1, p2, lw=1, col='blue'):
  ax.plot(*np.array([p1,p2]).T, color=col, linewidth=lw)

if __name__=='__main__':
  import matplotlib.pyplot as pyplot
  p1= np.random.uniform(-1,1,3)
  p2= np.random.uniform(-1,1,3)

  fig= pyplot.figure()
  ax= fig.add_subplot(111, projection='3d')
  PlotLine(ax, p1, p2)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim([-1.5,1.5])
  ax.set_ylim([-1.5,1.5])
  ax.set_zlim([-1.5,1.5])
  pyplot.show()
