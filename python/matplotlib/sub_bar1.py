#!/usr/bin/python3
#\file    sub_bar1.py
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
  fig, ax= plt.subplots()
  ax.bar(bar_loc+bar_offset[0], y_data['a'], bar_w, yerr=y_err['a'], label='a')
  ax.bar(bar_loc+bar_offset[1], y_data['b'], bar_w, yerr=y_err['b'], alpha=0.5, label='b')

  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_xticks(bar_loc)
  ax.set_xticklabels(x_data)
  ax.legend(loc='upper left')
  plt.show()
