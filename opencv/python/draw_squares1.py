#!/usr/bin/python
#\file    draw_squares1.py
#\brief   Draw squares.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.18, 2021
import numpy as np
import cv2

def GenSquarePattern1(img_w=200, img_h=200, w=10, h=50, N=10, scale=[0.8,1.2],
                      bg_col=(255,255,255), line_col=(2,48,155), fill_col=(86,168,228), with_reverse=False):
  unit_sq= np.array([[-w/2,-h/2],[-w/2,h/2],[w/2,h/2],[w/2,-h/2]])
  rot= lambda th: np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])

  pts_list= []
  for i in range(N):
    pts= (unit_sq*np.random.uniform(*scale)).dot(rot(np.random.uniform(0.0,np.pi)))+np.random.uniform([0.0,0.0],[img_w,img_h])
    pts= pts.astype(np.int32).reshape((-1,1,2))
    pts_list.append([pts])

  img= np.ones((img_h,img_w,3), np.uint8)*np.array(bg_col, np.uint8)
  for pts in pts_list:
    cv2.fillPoly(img, pts, fill_col)
    cv2.polylines(img, pts, True, line_col, 2)

  if with_reverse:
    imgr= np.ones((img_h,img_w,3), np.uint8)*np.array(bg_col, np.uint8)
    for pts in reversed(pts_list):
      cv2.fillPoly(imgr, pts, fill_col)
      cv2.polylines(imgr, pts, True, line_col, 2)
    return img,imgr
  return img

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

if __name__=='__main__':
  img,imgr= GenSquarePattern1(N=50,with_reverse=True)
  cv2.imshow('image',img)
  cv2.imshow('image(r)',imgr)
  while cv2.waitKey(100)&0xFF!=ord('q'):
    pass
  cv2.destroyAllWindows()
