#!/usr/bin/python
#\file    gen_sqptn1.py
#\brief   Generate square patterns.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.19, 2021
from opencv.draw_squares1 import *
import numpy as np
import cv2
import os

if __name__=='__main__':
  out_dir,i_start,i_end= 'data_generated/sqptn1/train',0,400
  #out_dir,i_start,i_end= 'data_generated/sqptn1/test',0,200

  try:
    os.makedirs(os.path.join(out_dir,'images'))
  except:
    pass
  try:
    os.makedirs(os.path.join(out_dir,'labels'))
  except:
    pass
  for i in range(i_start,i_end):
    density= np.random.uniform(0.0, 1.0)
    img= GenSquarePattern1(N=int(100*density))
    filename= '{0:06d}.jpg'.format(i)
    cv2.imwrite(os.path.join(out_dir,'images',filename),img)
    with open(os.path.join(out_dir,'labels',filename+'.dat'),'w') as fp:
      fp.write('{0}\n'.format(density))
    print 'Generated', filename
