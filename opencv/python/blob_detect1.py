#!/usr/bin/python
#\file    blob_detect1.py
#\brief   Test code of SimpleBlobDetector.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.05, 2023
import cv2
import numpy as np;

cap= cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

params= cv2.SimpleBlobDetector_Params()
detector= cv2.SimpleBlobDetector_create(params)

while(True):
  ret,frame= cap.read()

  keypoints= detector.detect(frame)

  disp_img= cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

  #cv2.imshow('camera',frame)
  cv2.imshow('blob',disp_img)
  if cv2.waitKey(1)&0xFF==ord('q'):
    break

cap.release()
cv2.destroyAllWindows()



