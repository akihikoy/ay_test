#!/usr/bin/python3
#\file    video_fps.py
#\brief   Get FPS of a video file.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.21, 2022

import cv2
import sys

if __name__=='__main__':
  video_file= sys.argv[1] if len(sys.argv)>1 else '../cpp/sample/vout2l.avi'
  vin= cv2.VideoCapture(video_file)
  if not vin.isOpened():
    raise Exception('Failed to open:',video_file)

  print('FPS from vin.get(CAP_PROP_FPS):', vin.get(getattr(cv2,'CV_CAP_PROP_FPS',getattr(cv2,'CAP_PROP_FPS',None))))
