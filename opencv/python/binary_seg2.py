#!/usr/bin/python
#\file    binary_seg2.py
#\brief   Segmentation of small binary image;
#         Simplified from binary_seg1.py
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.27, 2020
import numpy as np
import cv2
from matplotlib import pyplot as plt

def FindSegments(binary_img):
  # Finding sure foreground area
  dist_transform= cv2.distanceTransform(binary_img,cv2.cv.CV_DIST_L2,5)
  disp_img('dist_transform',dist_transform/10)
  ret,sure_fg= cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
  print 'sure_fg\n',np.uint8(sure_fg/255)
  disp_img('sure_fg',sure_fg)
  sure_fg= np.uint8(sure_fg)

  #ret,segments= cv2.connectedComponents(sure_fg)
  cnts,_= cv2.findContours(sure_fg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  print 'cnts\n',cnts
  segments= np.zeros(sure_fg.shape,np.int32)
  num_segments= len(cnts)
  for idx,cnt in enumerate(cnts):
    cv2.drawContours(segments, [cnt], -1, idx+1, -1, 8)
    cv2.drawContours(segments, [cnt], -1, idx+1, 2, 8)
  segments[binary_img==0]= 0
  print 'segments\n',segments,num_segments
  disp_img('segments',disp_int32(segments))
  return segments,num_segments


if __name__=='__main__':
  #img= cv2.imread('../cpp/sample/binary2.png')
  img= cv2.imread('../cpp/sample/binary3.png')
  #img= cv2.imread('../cpp/sample/binary4.png')
  #img= cv2.imread('../cpp/sample/binary5.png')
  print img.shape

  disp_int32= lambda img: np.uint8((img*2**8+(img>0)*2**10)/2**4)
  def disp_img(win_name, img, resize=(20,20)):
    cv2.imshow(win_name, cv2.resize(img,(img.shape[1]*resize[0],img.shape[0]*resize[1]),interpolation=cv2.INTER_NEAREST ))
    while cv2.waitKey() not in map(ord,[' ','q']):  pass

  disp_img('input', img*255)

  gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  segments,num_segments= FindSegments(gray)

  for idx in range(1,num_segments+1):
    img[segments==idx]= np.array([0,idx,0],np.uint8)

  disp_img('image',disp_int32(img))
