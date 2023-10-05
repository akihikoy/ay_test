#!/usr/bin/python
#\file    filter2d_test1_test.py
#\brief   Test of filter2d_test1 module.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.27, 2022
#NOTE: This program is incomplete (does not work).

import filter2d_test1
import cv2
import copy

if __name__=='__main__':
  cap= cv2.VideoCapture(0)
  #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
  #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
  f= filter2d_test1.TFilter2DTest1(10,100)

  while(True):
    ret,frame= cap.read()
    filtered= copy.deepcopy(frame)
    f.Apply(frame, filtered)
    '''
    TypeError: Apply(): incompatible function arguments. The following argument types are supported:
    1. (self: filter2d_test1.TFilter2DTest1, arg0: cv::_InputArray, arg1: cv::_OutputArray) -> None
    '''

    cv2.imshow('camera',filtered)
    if cv2.waitKey(1)&0xFF==ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
