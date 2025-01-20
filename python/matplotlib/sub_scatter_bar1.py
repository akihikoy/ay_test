#!/usr/bin/python3
#\file    sub_scatter_bar1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.16, 2021
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
  #plt.rcParams['figure.figsize']= 10,5
  fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

  X= np.linspace(0,5,20)
  ax1.scatter(X, 2.0*X+np.random.uniform(size=len(X)), color='blue', label='random')
  X= np.linspace(0,5,1000)
  ax1.plot(X, 2.0*X, color='blue', linewidth=3, label='linear')

  ax1.set_title('Test')
  ax1.set_xlabel('x')
  ax1.set_ylabel('y')
  ax1.legend()

  ax2.bar([1,2,3], [2,1,3], 0.2, yerr=[0.2,0.2,0.1], label='data')
  ax2.set_xlabel('x')
  ax2.set_ylabel('y')
  plt.show()
