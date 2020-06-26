#!/usr/bin/python
#\file    contour1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.16, 2020
import numpy as np
import six.moves.cPickle as pickle
import copy
import cv2
import scipy.ndimage
#import matplotlib.pyplot as plt

#Find multi-level contours in image img.
def FindMultilevelContours(img, vmin, vmax, step):
  contours= []
  #img= cv2.blur(copy.deepcopy(img),(3,3))
  for v in np.arange(vmin, vmax, step):
    img2= copy.deepcopy(img)
    img2[img<v]= 0
    img2[img>=v]= 1
    img2= img2.astype('uint8')
    #print img2.shape, img2.dtype
    cnts,_= cv2.findContours(img2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts)>0:  contours.append((v,cnts))
    #img2= cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(img2, cnts, -1, (0,0,255), 1)
    #cv2.imshow('debug',img2)
    #cv2.waitKey()
  return contours

#Visualize multi-level contours.
def DrawMultilevelContours(img, step=5):
  print 'min:',np.min(img)
  print 'max:',np.max(img)
  contours= FindMultilevelContours(img, np.min(img), np.max(img), step)
  img_viz= cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
  print len(contours)
  for v,cnts in contours:
    #col= (0,min(255,max(10,v-100)),0)
    col= (0,min(100,max(1,100-0.5*v)),0)
    thickness= 1
    cv2.drawContours(img_viz, cnts, -1, col, thickness)
  return img_viz

if __name__=='__main__':
  #img_depth= pickle.load(open('../../python/data/depth001.dat','rb'))['img_depth']
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/test_depth1.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/nprdepth001.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/nprdepth002.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/nprdepth003.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  print img_depth.shape, img_depth.dtype, [np.min(img_depth), np.max(img_depth)]

  #img_depth= scipy.ndimage.gaussian_filter(img_depth, sigma=7)
  #cv2.imshow('filtered',img_depth*155)


  img_viz= DrawMultilevelContours(img_depth)*255

  #data= img_depth.reshape(img_depth.shape[0],img_depth.shape[1])
  #plt.contour(data)
  #plt.show()

  cv2.imshow('depth',img_viz)
  #cv2.imshow('localmax',localmax)
  while cv2.waitKey() not in map(ord,[' ','q']):  pass

