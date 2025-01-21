#!/usr/bin/python3
#\file    aruco_detect2.py
#\brief   ArUco marker detection test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.29, 2024
import cv2
import numpy as np

cap= cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if __name__=='__main__':
  dictionary= cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
  parameters= cv2.aruco.DetectorParameters_create()

  Alpha= 1.0
  K= np.array([ 2.7276124573617790e+02, 0., 3.2933938280614751e+02, 0.,
              2.7301706435392521e+02, 2.4502942171375494e+02, 0., 0., 1. ]).reshape(3,3)
  D= np.array([ -2.7764926296767156e-01, 7.9998204984529891e-02,
          -8.3937489998474823e-04, 6.0999999999999943e-04,
          -7.4899999999988864e-03 ])
  size_in,size_out=  (640,480),(640,480)
  P,_= cv2.getOptimalNewCameraMatrix(K, D, size_in, Alpha, size_out)

  marker_length= 0.04  #size of the marker side in meters

  while(True):
    ret,frame= cap.read()

    corners, ids, rejectedImgPoints= cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    if ids is not None and len(ids)>0:
      cv2.aruco.drawDetectedMarkers(frame, corners, ids)
      #print 'corners:', corners
      rvecs, tvecs= cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, P, D)
      #draw axis for each marker
      for i in range(len(ids)):
        #cv2.drawFrameAxes(frame, P, D, rvecs[i], tvecs[i], length=0.05)  #For OpenCV 3.4+
        cv2.aruco.drawAxis(frame, P, D, rvecs[i], tvecs[i], length=0.05);

    cv2.imshow('marker_detection',frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
