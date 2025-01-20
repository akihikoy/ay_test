#!/usr/bin/python3
#\file    realtime1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.28, 2021
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

if __name__=='__main__':
  plt.rcParams['keymap.quit'].append('q')

  t= 0.0
  while True:
    plt.cla()
    X= np.linspace(t,t+5,20)
    plt.scatter(X, np.sin(X)+np.random.uniform(-0.2,0.2,size=len(X)), color='blue', label='random')
    X= np.linspace(t,t+5,1000)
    plt.plot(X, np.sin(X), color='red', linewidth=3, label='linear')
    t+= 0.1
    plt.title('Test')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.xlim(left=-3)
    plt.ylim(bottom=-1.2,top=1.2)
    plt.legend()
    plt.pause(0.05)

  #plt.show()
