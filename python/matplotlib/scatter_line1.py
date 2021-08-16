#!/usr/bin/python
#\file    scatter_line1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.16, 2021
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
  X= np.linspace(0,5,20)
  plt.scatter(X, 2.0*X+np.random.uniform(size=len(X)), color='blue', label='random')
  X= np.linspace(0,5,1000)
  plt.plot(X, 2.0*X, color='blue', linewidth=3, label='linear')

  plt.title('Test')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim(left=-3)
  plt.ylim(bottom=-3)
  plt.legend()
  plt.show()
