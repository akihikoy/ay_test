#!/usr/bin/python3
#\file    find_device_1.py
#\brief   Test of finding a video device (simple; just search for /dev/videoX).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.26, 2023

import glob
import cv2

if __name__=='__main__':
  video_devs= []
  for v_dev in glob.glob('/dev/video*'):
    cap= cv2.VideoCapture(v_dev)
    if cap.isOpened():
      video_devs.append(v_dev)
    cap.release()
  print('Video devices: {}'.format(video_devs))
