#!/usr/bin/python3
#\file    watershed1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.27, 2020
#Ref. https://docs.opencv.org/3.0.0/d3/db4/tutorial_py_watershed.html

import numpy as np
import cv2
from matplotlib import pyplot as plt

if __name__=='__main__':
  img = cv2.imread('../cpp/sample/water_coins.jpg')
  disp_int32 = lambda img: img*2**10+(img>0)*2**14
  def disp_img(win_name, img):
    cv2.imshow(win_name, img)
    while cv2.waitKey() & 0xFF not in map(ord,[' ','q']):  pass

  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  disp_img('thresh',thresh)

  # noise removal
  kernel = np.ones((3,3),np.uint8)
  opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
  disp_img('opening',opening)

  # sure background area
  sure_bg = cv2.dilate(opening,kernel,iterations=3)
  disp_img('sure_bg',sure_bg)

  # Finding sure foreground area
  #dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
  dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
  disp_img('dist_transform',dist_transform/10)
  ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
  disp_img('sure_fg',sure_fg)

  # Finding unknown region
  sure_fg = np.uint8(sure_fg)
  unknown = cv2.subtract(sure_bg,sure_fg)


  ## Marker labelling
  #ret, markers = cv2.connectedComponents(sure_fg)
  ## Add one to all labels so that sure background is not 0, but 1
  #markers = markers+1
  fcres= cv2.findContours(sure_fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  cnts= fcres[0] if len(fcres)==2 else fcres[1]
  print(cnts)
  markers = np.zeros(sure_fg.shape,np.int32)
  num_markers = len(cnts)
  for idx,cnt in enumerate(cnts):
    cv2.drawContours(markers, [cnt], -1, idx+1, -1, 8)
  markers = markers + 1
  disp_img('markers(contours)',disp_int32(markers))


  # Now, mark the region of unknown with zero
  markers[unknown==255] = 0
  print(markers)
  disp_img('markers(removing unknown)',disp_int32(markers))

  cv2.watershed(img,markers)
  print(markers)
  disp_img('markers(watershed)',disp_int32(markers))

  img[markers == -1] //= 2
  img[markers == -1] += np.array([128,0,0],np.uint8)
  for idx in range(0,num_markers+2):
    img[markers == idx] //= 2
    img[markers == idx] += np.array([0,idx*2,0],np.uint8)

  disp_img('image',img)

