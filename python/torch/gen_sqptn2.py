#!/usr/bin/python
#\file    gen_sqptn2.py
#\brief   Generate square patterns v2.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.03, 2021
from opencv.draw_squares1 import *
import numpy as np
import cv2
import os
import sys

if __name__=='__main__':
  mode= sys.argv[1] if len(sys.argv)>1 else 'train'
  a_size= float(sys.argv[2]) if len(sys.argv)>2 else 0.5
  if mode=='train':   i_start,i_end= 0,400
  elif mode=='test':  i_start,i_end= 0,200
  out_dir= 'data_generated/sqptn2/{}/{}'.format(a_size,mode)

  try:
    os.makedirs(os.path.join(out_dir,'input'))
  except:
    pass
  try:
    os.makedirs(os.path.join(out_dir,'output'))
  except:
    pass
  for i in range(i_start,i_end):
    density= np.random.uniform(a_size*0.5, 1.0-a_size*0.5)
    addition= np.random.uniform(-a_size*0.5, a_size*0.5)
    img= GenSquarePattern1(N=int(100*(density+addition)))
    filename= '{0:06d}'.format(i)
    cv2.imwrite(os.path.join(out_dir,'input',filename+'.jpg'),img)
    with open(os.path.join(out_dir,'input',filename+'.dat'),'w') as fp:
      fp.write('{0}\n'.format(density))
    with open(os.path.join(out_dir,'output',filename+'.dat'),'w') as fp:
      fp.write('{0}\n'.format(addition))
    print 'Generated', filename

