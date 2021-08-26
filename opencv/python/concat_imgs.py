#!/usr/bin/python
#\file    concat_imgs.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.26, 2021
import cv2
import numpy as np

if __name__=='__main__':
  img1= cv2.imread('../cpp/sample/rtrace1.png')
  img2= cv2.flip(img1, 0)

  cat_v= np.concatenate((img1,img2), axis=0)
  cat_h= np.concatenate((img1,img2), axis=1)
  cv2.imshow('concatenate vertically', cat_v)
  cv2.imshow('concatenate horizontally', cat_h)
  while cv2.waitKey() not in map(ord,[' ','q']):  pass
