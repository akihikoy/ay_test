#!/usr/bin/python3
#\file    cvtcolor.py
#\brief   Convert color
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.20, 2018
#src: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

import cv2

cap= cv2.VideoCapture(0)

while(True):
  # Capture frame-by-frame
  ret,frame= cap.read()

  # Our operations on the frame come here
  gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Display the resulting frame
  cv2.imshow('frame',gray)
  if cv2.waitKey(1)&0xFF==ord('q'):
    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
