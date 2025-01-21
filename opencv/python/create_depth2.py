#!/usr/bin/python3
#\file    create_depth2.py
#\brief   Generate a depth image for test(2).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.21, 2020
import numpy as np
import cv2

if __name__=='__main__':
  width,height= 640,480

  def f_depth1(x,y):
    return 250-x if 0<=x<250 else (x-250 if 250<=x<500 else 750-x)

  def f_depth2(x,y):
    if 0<=y<200:  return y if 0<=x<250 else (200-y if 250<=x<500 else 0.5*((x-500)+y))
    elif 200<=y:  return 250-x if 0<=x<250 else (x-250 if 250<=x<500 else 750-x)

  img= np.array([[[f_depth1(x,y)] for x in range(width)] for y in range(height)])
  img= img.astype(np.uint8)

  filename= '../cpp/sample/test_depth2.png'
  cv2.imwrite(filename,img)
  print('File is saved into:',filename)

  cv2.imshow('depth',img)
  while cv2.waitKey() not in list(map(ord,[' ','q'])):  pass

  img= np.array([[[f_depth2(x,y)] for x in range(width)] for y in range(height)])
  img= img.astype(np.uint8)

  filename= '../cpp/sample/test_depth3.png'
  cv2.imwrite(filename,img)
  print('File is saved into:',filename)

  cv2.imshow('depth',img)
  while cv2.waitKey() & 0xFF not in map(ord,[' ','q']):  pass
