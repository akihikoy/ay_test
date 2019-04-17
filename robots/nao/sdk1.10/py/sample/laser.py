#!/usr/bin/python
# -*- coding: utf-8 -*-
from naoqi import ALProxy
from naoconfig import *
import cv

proxyLRF = ALProxy('ALLaser',robot_IP,robot_port)
proxyLRF.laserON()

proxyMem = ALProxy('ALMemory',robot_IP,robot_port)

cv.NamedWindow("urg", 1)

width=500
height=500
scale=0.08
col_black=cv.Scalar(0,0,0)
col_blue=cv.Scalar(255,0,0)
col_red=cv.Scalar(0,0,255)
while True:
  print 'capture..'
  urgdata = proxyMem.getData('Device/Laser/Value')
  img = cv.CreateImage((width,height),cv.IPL_DEPTH_8U,3)
  cv.Rectangle(img,(0,0),(width-1,height-1),cv.Scalar(255,255,200),cv.CV_FILLED)
  cv.Circle(img,(width/2,height/2),10,col_blue,3)
  for p in urgdata:
    if p[0]==0:  continue;
    px = p[2]*scale
    py = p[3]*scale
    x = width/2+py
    y = height/2+px
    col = col_black
    if x>=width: x=width-1; col=col_red
    if x<0:      x=0; col=col_red
    if y>=height: y=height-1; col=col_red
    if y<0:       y=0; col=col_red
    cv.Circle(img,(x,y),3,col,cv.CV_FILLED)
  cv.ShowImage("urg", img)
  k = cv.WaitKey(10);
  if k==ord('q'):  break;

proxyLRF.laserOFF()
