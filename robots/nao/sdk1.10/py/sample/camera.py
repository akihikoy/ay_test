#!/usr/bin/python
# -*- coding: utf-8 -*-
from naoqi import ALProxy
from naoconfig import *
import cv

proxyCam = ALProxy('ALVideoDevice',robot_IP,robot_port)
#proxyCam.unsubscribe('my_GVM')

gvmName = 'my_GVM'
resolution = 1 # {0 = kQQVGA, 1 = kQVGA, 2 = kVGA}
colorSpace = 13 # {0 = kYuv, 9 = kYUV422, 10 = kYUV, 11 = kRGB, 12 = kHSY, 13 = kBGR}
fps = 15 # {5, 10, 15, 30}
nameId = proxyCam.subscribe(gvmName, resolution, colorSpace, fps)

cv.NamedWindow("camera", 1)

while True:
  print 'capture..'
  results = proxyCam.getImageRemote(nameId)
  width = results[0]
  height = results[1]
  img = cv.CreateImageHeader((width,height), cv.IPL_DEPTH_8U, 3)
  cv.SetData(img, results[6])
  cv.ShowImage("camera", img)
  k = cv.WaitKey(10);
  if k==ord('q'):  break;

proxyCam.unsubscribe(nameId)
