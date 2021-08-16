#!/usr/bin/python
#\file    bar1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.16, 2021
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
  groups= ['a','b']
  x_data= ['x1','x2','x3']
  y_data= {'a': [1,2,3],'b':[3,2,1]}
  y_err= {'a': [1,1,1],'b':[2,1,0.2]}

  bar_loc= np.arange(len(x_data))
  bar_w= 0.15
  bar_offset= [-bar_w/2,bar_w/2]
  plt.bar(bar_loc+bar_offset[0], y_data['a'], width=bar_w, yerr=y_err['a'], label='a')
  plt.bar(bar_loc+bar_offset[1], y_data['b'], width=bar_w, yerr=y_err['b'], alpha=0.5, label='b')

  plt.xlabel('x')
  plt.ylabel('y')
  plt.xticks(bar_loc, x_data)
  plt.legend(loc='upper left')
  plt.show()
