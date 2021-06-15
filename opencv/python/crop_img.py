#!/usr/bin/python
#\file    crop_img.py
#\brief   Crop an image.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.15, 2021

import numpy as np
import cv2
import sys,os

if __name__=='__main__':
  file_in= sys.argv[1]
  img= cv2.imread(file_in)
  print 'Input image shape:',img.shape

  x,y,w,h= 260,180,120,300
  img_cropped= img[y:y+h,x:x+w]
  #img_cropped= cv2.flip(cv2.transpose(img_cropped),1)
  print 'Cropped image shape:',img_cropped.shape

  cv2.imshow('image', img)
  if 0 not in img_cropped.shape:  cv2.imshow('cropped', img_cropped)
  while True:
    key= cv2.waitKey()
    if key in map(ord,[' ','q']):  break
    if key==ord('s'):
      file_out= os.path.basename(file_in)
      if not os.path.exists(file_out):
        cv2.imwrite(file_out,img_cropped)
        print 'Saved the image to the file:',file_out
      else:
        print 'Failed to save the image as the file already exists:',file_out


