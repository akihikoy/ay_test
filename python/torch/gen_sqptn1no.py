#!/usr/bin/python
#\file    gen_sqptn1no.py
#\brief   Generate square patterns (no overlap, exact density).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.22, 2021
from opencv.draw_squares2 import *
import numpy as np
import cv2
import os
import sys

if __name__=='__main__':
  mode= sys.argv[1] if len(sys.argv)>1 else 'train'
  if mode=='train':   out_dir,i_start,i_end= 'data_generated/sqptn1no/train',0,4000
  elif mode=='test':  out_dir,i_start,i_end= 'data_generated/sqptn1no/test',0,2000

  try:
    os.makedirs(os.path.join(out_dir,'images'))
  except:
    pass
  try:
    os.makedirs(os.path.join(out_dir,'labels'))
  except:
    pass
  for i in range(i_start,i_end):
    density= round(100.*np.random.uniform(0.0, 1.0))/100.
    img= GenSquarePattern2(N=int(100*density))
    filename,filenamer= '{0:06d}.jpg'.format(i),'{0:06d}r.jpg'.format(i)
    cv2.imwrite(os.path.join(out_dir,'images',filename),img)
    with open(os.path.join(out_dir,'labels',filename+'.dat'),'w') as fp:
      fp.write('{0}\n'.format(density))
    print 'Generated', filename
