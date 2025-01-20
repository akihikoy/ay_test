#!/usr/bin/python3
#\file    filled_yerr1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.31, 2021
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
  X= np.linspace(0,5,1000)
  plt.fill_between(X, (2.0*X*X-X).reshape(-1), (2.0*X*X+X).reshape(-1), alpha=0.5)
  plt.plot(X, 2.0*X*X, color='blue', linewidth=3, label='quad')
  X= np.linspace(0,5,20)
  plt.scatter(X, 2.0*X*X+X*np.random.uniform(-1,1,size=len(X)), color='red', label='random')

  plt.title('Test')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend(loc='upper left')
  plt.rcParams['keymap.quit'].append('q')
  plt.show()
