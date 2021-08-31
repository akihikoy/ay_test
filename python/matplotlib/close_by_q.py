#!/usr/bin/python
#\file    close_by_q.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.31, 2021
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
  X= np.linspace(0,5,20)
  plt.scatter(X, 2.0*X+np.random.uniform(size=len(X)), color='blue', label='random')

  plt.title('Close by pressing q')
  print 'keymap.quit:', plt.rcParams['keymap.quit']
  plt.rcParams['keymap.quit'].append('q')
  plt.show()
