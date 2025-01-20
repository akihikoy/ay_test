#!/usr/bin/python3
#\file    sub_line1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.16, 2021
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

if __name__=='__main__':
  fig= plt.figure()
  ax= fig.add_subplot(1,1,1)

  x= np.linspace(-6,6,1000)

  ax.plot(x,norm.pdf(x, loc=0.0, scale=1.0), color='black',  linestyle='solid')
  ax.plot(x,norm.pdf(x, loc=0.0, scale=0.5), color='black',  linestyle='dashed')
  ax.plot(x,norm.pdf(x, loc=0.0, scale=0.25), color='black', linestyle='dashdot')

  ax.set_title('First line plot')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.grid(True)
  plt.show()
