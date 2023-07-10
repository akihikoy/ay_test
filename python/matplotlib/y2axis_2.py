#!/usr/bin/python
#\file    y2axis_2.py
#\brief   Similar to y2axis, but put a single legend box.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.10, 2023
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

if __name__=='__main__':
  fig= plt.figure()
  ax1= fig.add_subplot(1,1,1)

  lines= []
  x= np.linspace(-6,6,1000)
  lines+= ax1.plot(x,norm.pdf(x, loc=0.0, scale=1.0), color='red', linestyle='solid', label='SD=1')
  lines+= ax1.plot(x,norm.pdf(x, loc=0.0, scale=2.0), color='red', linestyle='dashed', label='SD=2.0')
  ax1.set_title('First line plot')
  ax1.set_xlabel('x')
  ax1.set_ylabel('y')
  #ax1.grid(True)

  ax2= ax1.twinx()
  lines+= ax2.plot(x,norm.pdf(x, loc=0.0, scale=2.0), color='blue', linestyle='solid', label='SD=2.0')
  ax2.set_ylabel('y2')

  ax2.legend(lines, [l.get_label() for l in lines], loc='upper left')

  plt.show()
