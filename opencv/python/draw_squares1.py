#!/usr/bin/python
#\file    draw_squares1.py
#\brief   Draw squares.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.18, 2021
import numpy as np
import cv2

def GenSquarePattern1(img_w=200, img_h=200, w=10, h=50, N=10, scale= [0.8,1.2]):
  #w,h= 10,50
  unit_sq= np.array([[-w/2,-h/2],[-w/2,h/2],[w/2,h/2],[w/2,-h/2]])
  #scale= [0.8,1.2]
  #N= 10

  img= np.ones((img_h,img_w,3), np.uint8)*np.array([255,255,255], np.uint8)

  rot= lambda th: np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])

  for i in range(N):
    pts= (unit_sq*np.random.uniform(*scale)).dot(rot(np.random.uniform(0.0,np.pi)))+np.random.uniform([0.0,0.0],[img_w,img_h])
    pts= pts.astype(np.int32).reshape((-1,1,2))
    cv2.fillPoly(img, [pts], (86,168,228))
    cv2.polylines(img, [pts], True, (2,48,155), 2)

  #sqs= []
  #for i in range(N):
    #pts= unit_sq.dot(rot(np.random.uniform(0.0,np.pi)))+np.random.uniform([0.0,0.0],[img_w,img_h])
    #pts= pts.astype(np.int32).reshape((-1,1,2))
    #sqs.append(pts)
  #cv2.fillPoly(img, sqs, (255,0,255))
  #cv2.polylines(img, sqs, True, (128,0,128), 2)

  #sqs= [(unit_sq.dot(rot(np.random.uniform(0.0,np.pi)))+np.random.uniform([0.0,0.0],[img_w,img_h])).astype(np.int32).reshape((-1,1,2)) for i in range(N)]
  #cv2.fillPoly(img, sqs, (255,0,255))
  #cv2.polylines(img, sqs, True, (128,0,128), 2)

  return img

if __name__=='__main__':
  img= GenSquarePattern1(N=50)
  cv2.imshow('image',img)
  while cv2.waitKey(100)&0xFF!=ord('q'):
    pass
  cv2.destroyAllWindows()
