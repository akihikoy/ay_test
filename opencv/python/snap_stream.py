#!/usr/bin/python
#\file    snap_stream.py
#\brief   Take snapshots and store them to a directory.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.17, 2021

import cv2
import sys
dest_dir= sys.argv[1]
frame_skip= int(sys.argv[2]) if len(sys.argv)>2 else 5

cap= cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

save_mode= False
for i_frame in xrange(1000000):
  ret,frame= cap.read()
  if save_mode and i_frame%frame_skip==0:
    filename= '{}/{:05d}.jpg'.format(dest_dir,i_frame)
    cv2.imwrite(filename, frame)
    print 'saved:',filename

  cv2.imshow('camera',frame)
  key= cv2.waitKey(1)&0xFF
  if key==ord('q'):  break
  elif key==ord(' '):  save_mode= not save_mode

cap.release()
cv2.destroyAllWindows()
