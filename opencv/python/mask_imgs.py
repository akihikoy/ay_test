#!/usr/bin/python
#\file    mask_imgs.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.08, 2021
import numpy as np
import cv2

if __name__=='__main__':
  img1= cv2.imread('../cpp/sample/banana-peel_620x350_71497523358.jpg')
  img2= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.uint8)
  img3= 1.0-img2/255.0
  print 'shapes:',img1.shape,img2.shape,img3.shape

  def apply_mask(img):
    points= np.array([[100,100],[100,200],[200,200],[200,100]])
    mask_img= np.zeros(img.shape[:2], dtype='uint8')
    cv2.fillPoly(mask_img, [points.reshape(-1,1,2)], 1)
    mask= np.ones_like(mask_img, dtype=bool)
    mask[mask_img>0]= False
    if len(img.shape)==3:  mask= np.repeat(mask[:,:,np.newaxis],img.shape[2],axis=2)
    masked= img.copy()
    masked[mask]= 0
    return masked

  cv2.imshow('win1', apply_mask(img1))
  cv2.imshow('win2', apply_mask(img2))
  cv2.imshow('win3', apply_mask(img3))
  while cv2.waitKey() not in map(ord,[' ','q']):  pass
