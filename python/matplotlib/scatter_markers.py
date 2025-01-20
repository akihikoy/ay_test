#!/usr/bin/python3
#\file    scatter_markers.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.31, 2021
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
  markers= ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
  print('# of markers:', len(markers))
  X= np.linspace(-2,2,20)

  plt.rcParams['keymap.quit'].append('q')
  for i in range(12):
    plt.scatter(X, 1./(i-6.5)*X, marker=markers[i], s=64, color=plt.get_cmap('Set1')(i*12), label='Marker {}'.format(markers[i]))
  plt.title('Marker test')
  plt.xlim(-2,3)
  plt.ylim(-2,2)
  plt.legend()
  plt.show()

  for i in range(12,len(markers)):
    plt.scatter(X, 1./(i-12-6.5)*X, marker=markers[i], s=64, color=plt.get_cmap('Set1')(i*12), label='Marker {}'.format(markers[i]))
  plt.title('Marker test')
  plt.xlim(-2,3)
  plt.ylim(-2,2)
  plt.legend()
  plt.show()
