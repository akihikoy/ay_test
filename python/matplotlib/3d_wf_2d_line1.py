#!/usr/bin/python3
#\file    3d_wf_2d_line1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.16, 2021
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

if __name__=='__main__':
  xy= np.mgrid[-2:2:0.1,-2:2:0.1]
  fig= plt.figure(figsize=(10,5))
  ax1= fig.add_subplot(1,2,1,projection='3d')
  ax1.plot_wireframe(xy[0], xy[1], xy[0]**2+xy[1]**2, color=[0,1,0])

  ax2= fig.add_subplot(1,2,2)
  x= np.linspace(-6,6,1000)
  ax2.plot(x, np.sin(x), color='red', linestyle='dotted')

  ax1.set_xlabel('x')
  ax1.set_ylabel('y')
  ax1.set_zlabel('z')

  plt.show()
