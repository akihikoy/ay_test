#!/usr/bin/python3
#\file    sub_scatter_line1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.16, 2021
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
  fig, ax = plt.subplots()

  X= np.linspace(0,5,20)
  ax.scatter(X, 2.0*X+np.random.uniform(size=len(X)), color='blue', label='random')
  X= np.linspace(0,5,1000)
  ax.plot(X, 2.0*X, color='blue', linewidth=3, label='linear')

  ax.set_title('Test')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_xlim(left=-3)
  ax.set_ylim(bottom=-3)
  ax.legend()
  plt.show()
