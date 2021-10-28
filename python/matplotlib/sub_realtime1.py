#!/usr/bin/python
#\file    sub_realtime1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.28, 2021
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
  #plt.rcParams['figure.figsize']= 10,5
  fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

  t= 0.0
  while True:
    ax1.cla()
    X= np.linspace(t,t+5,20)
    ax1.scatter(X, np.sin(X)+np.random.uniform(-0.2,0.2,size=len(X)), color='blue', label='random')
    X= np.linspace(t,t+5,1000)
    ax1.plot(X, np.sin(X), color='red', linewidth=3, label='linear')

    ax2.cla()
    X= np.linspace(0,np.pi*4.0,1000)
    ax2.plot(np.cos(t+2.0*X), np.sin(t+3.0*X), color='green', linewidth=3, label='linear')

    t+= 0.1
    ax1.set_title('Test')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_ylim(bottom=-1.2,top=1.2)
    ax1.legend()
    ax2.set_title('Test2')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.pause(0.05)

  #plt.show()
