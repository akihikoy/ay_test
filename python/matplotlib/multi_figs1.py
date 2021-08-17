#!/usr/bin/python
#\file    multi_figs1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.17, 2021
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
  fig1= plt.figure()
  ax= fig1.add_subplot(1,1,1)
  x= np.linspace(-6,6,1000)
  ax.plot(x, np.sin(x), color='blue',  linestyle='solid')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.grid(True)

  fig2= plt.figure(figsize=(8,8))
  rows,cols= 5,4
  for i in range(rows*cols):
    img= np.random.randint(256,size=(10,10,3),dtype=np.uint8)
    ax= fig2.add_subplot(rows, cols, i+1)
    ax.set_title('image {0}'.format(i), fontsize=10)
    ax.imshow(img)
  fig2.tight_layout()

  plt.show()
