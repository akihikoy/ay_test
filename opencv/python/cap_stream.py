#!/usr/bin/python3
#\file    cap_stream.py
#\brief   Capture video from an html video stream.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.20, 2018
'''
To test this code, generate a stream with:
~/prg/mjpg-streamer2/mjpg-streamer-experimental$ ./mjpg_streamer -i "./input_uvc.so -f 10 -r 320x240 -d /dev/video0 -y -n" -o "./output_http.so -w ./www -p 8080"
'''


import cv2

cap= cv2.VideoCapture('http://localhost:8080/?action=stream&dummy=file.mjpg')

while(True):
  ret,frame= cap.read()

  cv2.imshow('camera',frame)
  if cv2.waitKey(1)&0xFF==ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
