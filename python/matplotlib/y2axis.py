#!/usr/bin/python
#\file    y2axis.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.01, 2021
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

if __name__=='__main__':
  fig= plt.figure()
  ax1= fig.add_subplot(1,1,1)

  x= np.linspace(-6,6,1000)
  ax1.plot(x,norm.pdf(x, loc=0.0, scale=1.0), color='red', linestyle='solid', label='SD=1')
  ax1.plot(x,norm.pdf(x, loc=0.0, scale=2.0), color='red', linestyle='dashed', label='SD=2.0')
  ax1.set_title('First line plot')
  ax1.set_xlabel('x')
  ax1.set_ylabel('y')
  #ax1.grid(True)
  ax1.legend()

  ax2= ax1.twinx()
  ax2.plot(x,norm.pdf(x, loc=0.0, scale=2.0), color='blue', linestyle='solid', label='SD=2.0')
  ax2.set_ylabel('y2')
  ax2.legend(loc='upper right', bbox_to_anchor=(1.0,0.8))

  plt.show()
