#!/usr/bin/python
#\file    resize.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.14, 2019
import numpy as np
import six.moves.cPickle as pickle
import cv2

if __name__=='__main__':
  img= pickle.load(open('../../python/data/depth001.dat','rb'))['img_depth']
  #print img_depth.shape

  resize_ratio= 0.5
  img= cv2.resize(img,tuple(map(int,np.flipud(img.shape[:2])*resize_ratio)),interpolation=cv2.INTER_NEAREST)

  cv2.imshow('image',img*255)
  #cv2.imshow('localmax',localmax)
  while cv2.waitKey() not in map(ord,[' ','q']):  pass

