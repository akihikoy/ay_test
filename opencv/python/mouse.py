#!/usr/bin/python3
#\file    mouse.py
#\brief   Detect mouse event.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.20, 2018
import cv2

def OnMouse(event, x, y, flags, param):
  if event==cv2.EVENT_LBUTTONUP:
    print('LBUTTONUP',x,y)
  elif event==cv2.EVENT_RBUTTONUP and flags&cv2.EVENT_FLAG_SHIFTKEY:
    print('RBUTTONUP+SHIFTKEY',x,y)
  else:
    print(event,x,y,flags,param)

cv2.namedWindow("camera")
cv2.setMouseCallback("camera", OnMouse)

cap= cv2.VideoCapture(0)

while(True):
  ret,frame= cap.read()

  cv2.imshow('camera',frame)
  if cv2.waitKey(1)&0xFF==ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
