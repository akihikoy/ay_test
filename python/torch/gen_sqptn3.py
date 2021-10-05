#!/usr/bin/python
#\file    gen_sqptn3.py
#\brief   Generate square patterns v3.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.15, 2021
from opencv.draw_squares1 import *
import numpy as np
import cv2
import os
import sys

if __name__=='__main__':
  mode= sys.argv[1] if len(sys.argv)>1 else 'train'
  a_size1= float(sys.argv[2]) if len(sys.argv)>2 else 0.3
  a_size2= float(sys.argv[3]) if len(sys.argv)>3 else 0.3
  if mode=='train':   i_start,i_end= 0,400
  elif mode=='test':  i_start,i_end= 0,200
  out_dir= 'data_generated/sqptn3/{}_{}/{}'.format(a_size1,a_size2,mode)
  #if mode=='train':   i_start,i_end= 0,800
  #elif mode=='test':  i_start,i_end= 0,400
  #out_dir= 'data_generated/sqptn3l/{}_{}/{}'.format(a_size1,a_size2,mode)

  try:
    os.makedirs(os.path.join(out_dir,'input'))
  except:
    pass
  try:
    os.makedirs(os.path.join(out_dir,'output'))
  except:
    pass
  for i in range(i_start,i_end):
    density= np.random.uniform((a_size1+a_size2)*0.5, 1.0-(a_size1+a_size2)*0.5)
    addition1= np.random.uniform(-a_size1*0.5, a_size1*0.5)
    addition2= np.random.uniform(-a_size2*0.5, a_size2*0.5)
    img1= GenSquarePattern1(N=int(100*(density+addition1)))
    img2= GenSquarePattern1(N=int(100*(density+addition2)))
    filename= '{0:06d}'.format(i)
    cv2.imwrite(os.path.join(out_dir,'input',filename+'-1.jpg'),img1)
    cv2.imwrite(os.path.join(out_dir,'input',filename+'-2.jpg'),img2)
    with open(os.path.join(out_dir,'input',filename+'.dat'),'w') as fp:
      fp.write('{0}\n'.format(density))
    with open(os.path.join(out_dir,'output',filename+'.dat'),'w') as fp:
      fp.write('{0}\n'.format(addition1+addition2))
    print 'Generated', filename

