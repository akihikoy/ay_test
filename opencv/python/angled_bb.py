#!/usr/bin/python3
#\file    angled_bb.py
#\brief   Get a bounding box aligned at a given angle.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.29, 2020
import numpy as np
import cv2


#Get a bounding box aligned at a given angle.
def GetAlignedBoundingBox(points, angle):
  c,s= np.cos(angle), np.sin(angle)
  R= np.array(((c, -s), (s, c)))
  aabb_center= np.mean(points,axis=0)
  points_aligned= np.dot(np.array(points) - aabb_center, R)
  aabb_w,aabb_h= np.max(points_aligned,axis=0)-np.min(points_aligned,axis=0)
  return aabb_center,aabb_w,aabb_h


if __name__=='__main__':
  w,h= np.random.randint(400), np.random.randint(300)
  left,top= np.random.randint(640-w), np.random.randint(480-h)
  points= [[left+np.random.randint(w),top+np.random.randint(h)] for _ in range(30)]

  angle= np.random.uniform(-np.pi,np.pi)

  aabb_center,aabb_w,aabb_h= GetAlignedBoundingBox(points, angle)
  print(aabb_center,aabb_w,aabb_h)

  img= np.zeros((480,640,3), np.uint8)
  for pt in points:
    cv2.circle(img, tuple(pt), 10, (0,255,0), 1)

  c,s= np.cos(angle), np.sin(angle)
  R= np.array(((c, -s), (s, c)))
  cv2.circle(img, tuple(np.int32(aabb_center)), 20, (255,0,0), 1)
  cv2.polylines(img, np.array([[aabb_center,aabb_center+np.dot([50,0],R.T)]],np.int32), True, (255,0,0), 2)
  corners= np.array([[-aabb_w/2.0,-aabb_h/2.0],[aabb_w/2.0,-aabb_h/2.0],[aabb_w/2.0,aabb_h/2.0],[-aabb_w/2.0,aabb_h/2.0]])
  pts_corner= aabb_center + np.dot(corners,R.T)
  cv2.polylines(img, [np.int32(pts_corner)], True, (0,0,255), 2)


  cv2.imshow('image',img)
  while cv2.waitKey() & 0xFF not in map(ord,[' ','q']):  pass
