#!/usr/bin/python
#\file    cap_cam.py
#\brief   Capture video from a camera.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.20, 2018

import cv2

cap= cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while(True):
  ret,frame= cap.read()

  cv2.imshow('camera',frame)
  if cv2.waitKey(1)&0xFF==ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
