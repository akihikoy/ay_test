#!/usr/bin/python3
#DEPRECATED: This is an old fashioned code.
# -*- coding: utf-8 -*-
# src: http://d.hatena.ne.jp/kuri27/20110523/1306099283
# DEPEND: python-opencv
import cv
import time

cv.NamedWindow('camera', 1)
capture= cv.CreateCameraCapture(0)

width= None #leave None for auto-detection
height= None #leave None for auto-detection

if width is None:
  width= int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH))
else:
  cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_WIDTH,width)

if height is None:
  height= int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT))
else:
  cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_HEIGHT,height)

while True:
  img= cv.QueryFrame(capture)
  cv.ShowImage('camera', img)
  k= cv.WaitKey(10) & 0xFF;
  if k==ord('q'):
    break
