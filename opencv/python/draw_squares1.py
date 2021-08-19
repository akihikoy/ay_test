#!/usr/bin/python
#\file    draw_squares1.py
#\brief   Draw squares.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.18, 2021
import numpy as np
import cv2

w,h= 10,50
unit_sq= np.array([[-w/2,-h/2],[-w/2,h/2],[w/2,h/2],[w/2,-h/2]])
scale= [0.8,1.2]
N= 10

img= np.ones((200,200,3), np.uint8)*np.array([255,255,255], np.uint8)

rot= lambda th: np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])

for i in range(N):
  pts= (unit_sq*np.random.uniform(*scale)).dot(rot(np.random.uniform(0.0,np.pi)))+np.random.uniform([0.0,0.0],[200.0,200.0])
  pts= pts.astype(np.int32).reshape((-1,1,2))
  cv2.fillPoly(img, [pts], (86,168,228))
  cv2.polylines(img, [pts], True, (2,48,155), 2)

#sqs= []
#for i in range(N):
  #pts= unit_sq.dot(rot(np.random.uniform(0.0,np.pi)))+np.random.uniform([0.0,0.0],[200.0,200.0])
  #pts= pts.astype(np.int32).reshape((-1,1,2))
  #sqs.append(pts)
#cv2.fillPoly(img, sqs, (255,0,255))
#cv2.polylines(img, sqs, True, (128,0,128), 2)

#sqs= [(unit_sq.dot(rot(np.random.uniform(0.0,np.pi)))+np.random.uniform([0.0,0.0],[200.0,200.0])).astype(np.int32).reshape((-1,1,2)) for i in range(N)]
#cv2.fillPoly(img, sqs, (255,0,255))
#cv2.polylines(img, sqs, True, (128,0,128), 2)

cv2.imshow('image',img)
while cv2.waitKey(100)&0xFF!=ord('q'):
  pass
cv2.destroyAllWindows()
