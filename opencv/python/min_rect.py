#!/usr/bin/python
#\file    min_rect.py
#\brief   Test of minAreaRect.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.16, 2024
import numpy as np
import cv2
from cv2 import minAreaRect as cv2_minAreaRect

##Return an angled bounding box parameters for given points.
##  return: (aabb_center_x,aabb_center_y),(aabb_w,aabb_h),aabb_angle
##  NOTE: aabb_w >= aabb_h, aabb_angle in [-pi/2,pi/2].
#def GetAngledBoundingBox(points):
  #min_rect= cv2.minAreaRect(np.array(points))
  #(aabb_center_x,aabb_center_y),(aabb_w,aabb_h),aabb_angle= min_rect
  #aabb_angle= aabb_angle/180.0*np.pi
  #if aabb_w<aabb_h:
    #aabb_w,aabb_h= aabb_h,aabb_w
    #aabb_angle= aabb_angle+0.5*np.pi
  #if aabb_angle>np.pi:  aabb_angle-= np.pi
  #if aabb_angle<-np.pi:  aabb_angle+= np.pi
  #return (aabb_center_x,aabb_center_y),(aabb_w,aabb_h),aabb_angle

#Get a rectangle of minimum area.
#Return: center,size,angle.
#  angle is in radian.
#  size[0] is always greater than size[1].
#ref. https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points
def MinAreaRect(points):
  center,size,angle= cv2_minAreaRect(np.array(points,np.float32))
  angle*= np.pi/180.0
  if size[0]<size[1]:
    size= (size[1],size[0])
    angle= angle+np.pi*0.5 if angle<0 else angle-np.pi*0.5
  return center,size,angle

if __name__=='__main__':
  w,h= np.random.randint(400), np.random.randint(300)
  left,top= np.random.randint(640-w), np.random.randint(480-h)
  points= [[left+np.random.randint(w),top+np.random.randint(h)] for _ in range(30)]
  #points= [(lambda th:[int(300+150*np.cos(th)),int(200+150*np.sin(th))])(np.random.randint(360)) for _ in range(30)]

  #aabb_center,(aabb_w,aabb_h),aabb_angle= GetAngledBoundingBox(points)
  aabb_center,(aabb_w,aabb_h),aabb_angle= MinAreaRect(points)

  #print min_rect, type(min_rect)
  print aabb_center,aabb_w,aabb_h,aabb_angle

  img= np.zeros((480,640,3), np.uint8)
  for pt in points:
    cv2.circle(img, tuple(pt), 3, (0,255,0), 1)

  c,s= np.cos(aabb_angle), np.sin(aabb_angle)
  R= np.array(((c, -s), (s, c)))
  cv2.circle(img, tuple(np.int32(aabb_center)), 10, (255,0,0), 1)
  cv2.polylines(img, np.array([[aabb_center,aabb_center+np.dot([50,0],R.T)]],np.int32), True, (255,0,0), 2)
  corners= np.array([[-aabb_w/2.0,-aabb_h/2.0],[aabb_w/2.0,-aabb_h/2.0],[aabb_w/2.0,aabb_h/2.0],[-aabb_w/2.0,aabb_h/2.0]])
  pts_corner= aabb_center + np.dot(corners,R.T)
  cv2.polylines(img, [np.int32(pts_corner)], True, (0,0,255), 2)


  cv2.imshow('image',img)
  while cv2.waitKey() not in map(ord,[' ','q']):  pass

