#!/usr/bin/python
#\file    mouse_poly.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.21, 2018
import numpy as np
import cv2

def OnMouse(event, x, y, flags, param):
  if event==cv2.EVENT_LBUTTONUP:
    poly.append([x,y])
  elif event==cv2.EVENT_RBUTTONUP:
    print poly
    while len(poly):  poly.pop()

poly= []
cv2.namedWindow("camera")
cv2.setMouseCallback("camera", OnMouse, param=poly)

cap= cv2.VideoCapture(0)

while(True):
  ret,frame= cap.read()

  if len(poly)>0:
    if len(poly)>1:
      pts= np.array(poly, np.int32).reshape((-1,1,2))
    else:
      pts= np.array([poly[0],poly[0],poly[0]], np.int32).reshape((-1,1,2))
    cv2.polylines(frame, [pts], True, (255,0,128), 5)

  cv2.imshow('camera',frame)
  if cv2.waitKey(1)&0xFF==ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
