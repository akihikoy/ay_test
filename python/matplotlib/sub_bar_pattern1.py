#!/usr/bin/python
#\file    sub_bar_pattern1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.16, 2021
import numpy as np
import matplotlib.pyplot as plt
import itertools

if __name__=='__main__':
  plt.rcParams['figure.figsize']= 15,5
  groups= ['a','b']
  x_data= ['x1','x2','x3']
  y_data= {'a': [1,2,3],'b':[3,2,1]}

  bar_loc= np.arange(len(x_data))
  bar_w= 0.15
  bar_offset= [-bar_w/2,bar_w/2]
  fig, (ax1,ax2,ax3) = plt.subplots(1,3)

  patterns1= ('-','+','x','\\','*','o','O','.')
  patterns2= ('/','\\')
  patterns3= ('///','\\\\\\')

  bars= ax1.bar(bar_loc+bar_offset[0], y_data['a'], bar_w, label='a')
  bars+= ax1.bar(bar_loc+bar_offset[1], y_data['b'], bar_w, alpha=0.5, label='b')
  for bar, pattern in zip(bars, itertools.cycle(patterns1)):
      bar.set_hatch(pattern)

  bars= ax2.bar(bar_loc+bar_offset[0], y_data['a'], bar_w, label='a')
  for bar in bars:  bar.set_hatch(patterns2[0])
  bars= ax2.bar(bar_loc+bar_offset[1], y_data['b'], bar_w, alpha=0.5, label='b')
  for bar in bars:  bar.set_hatch(patterns2[1])

  bars= ax3.bar(bar_loc+bar_offset[0], y_data['a'], bar_w, label='a')
  for i,bar in enumerate(bars):  bar.set_hatch(patterns3[i%len(patterns3)])
  bars= ax3.bar(bar_loc+bar_offset[1], y_data['b'], bar_w, alpha=0.5, label='b')
  for i,bar in enumerate(bars):  bar.set_hatch(patterns3[i%len(patterns3)])

  for ax in (ax1,ax2,ax3):
      ax.set_xlabel('x')
      ax.set_ylabel('y')
      ax.set_xticks(bar_loc, x_data)
      ax.legend(loc='upper left')
  plt.show()
