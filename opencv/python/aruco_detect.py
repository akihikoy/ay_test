#!/usr/bin/python3
#\file    aruco_detect.py
#\brief   ArUco marker detection test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.28, 2024
import cv2

cap= cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if __name__=='__main__':
  dictionary= cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
  parameters= cv2.aruco.DetectorParameters_create()

  while(True):
    ret,frame= cap.read()

    corners, ids, rejectedImgPoints= cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    print('corners:', corners)

    cv2.imshow('marker_detection',frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
