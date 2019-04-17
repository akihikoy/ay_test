#!/usr/bin/python
#\file    cap_stream.py
#\brief   Capture video from an html video stream.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.20, 2018

import cv2

cap= cv2.VideoCapture('http://aypi10:8080/?action=stream&dummy=file.mjpg')

while(True):
  ret,frame= cap.read()

  cv2.imshow('camera',frame)
  if cv2.waitKey(1)&0xFF==ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
