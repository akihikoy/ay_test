#!/usr/bin/python3
#\file    3d_wireframe2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.16, 2021
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

if __name__=='__main__':
  xy= np.mgrid[-2:2:0.1,-2:2:0.1]
  fig= plt.figure()
  ax= fig.add_subplot(1,1,1,projection='3d')
  ax.plot_wireframe(xy[0], xy[1], xy[0]**2+xy[1]**2, color=[0,1,0])

  xy2= np.random.uniform(-2,2,(2,100))
  ax.scatter(xy2[0], xy2[1], xy2[0]**2+xy2[1]**2, marker='*', color=[1,0,0])

  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.show()
