#!/usr/bin/python3
#\file    opencv_version.py
#\brief   Print the OpenCV version.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.21, 2022
import cv2

if __name__=='__main__':
  print('OpenCV version : ',   cv2.__version__)
  print('Major version : ',    cv2.__version__.split('.')[0])
  print('Minor version : ',    cv2.__version__.split('.')[1])
  print('Subminor version : ', cv2.__version__.split('.')[2])
