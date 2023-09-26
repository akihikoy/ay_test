#!/usr/bin/python
#\file    filter2d_test2_test.py
#\brief   Test of filter2d_test2.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.26, 2023
import filter2d_test2
import cv2
import copy
import numpy as np

if __name__=='__main__':
  cap= cv2.VideoCapture(0)
  #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
  #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
  filter2d_test2.MakeKernel(10,30)

  while(True):
    ret,frame= cap.read()
    if frame is None:  continue

    filtered= np.empty_like(frame)
    filtered= filter2d_test2.ApplyFilter(frame)

    cv2.imshow('camera',frame)
    cv2.imshow('filtered',filtered)
    if cv2.waitKey(1)&0xFF==ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
