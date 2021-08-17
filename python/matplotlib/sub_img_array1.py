#!/usr/bin/python
#\file    sub_img_array1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.17, 2021
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
  fig= plt.figure(figsize=(8,8))
  rows,cols= 5,4
  for i in range(rows*cols):
    img= np.random.randint(256,size=(10,10,3),dtype=np.uint8)
    ax= fig.add_subplot(rows, cols, i+1)
    ax.set_title('image {0}'.format(i), fontsize=10)
    ax.imshow(img)
  fig.tight_layout()
  plt.show()
