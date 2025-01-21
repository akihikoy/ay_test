#!/usr/bin/python3
#\file    aruco_create.py
#\brief   Creating an ArUco marker.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.28, 2024
import cv2
import sys

if __name__=='__main__':
  ID= int(sys.argv[1]) if len(sys.argv)>1 else 1
  size= 200
  borderBits= 1

  dictionary= cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
  img= cv2.aruco.drawMarker(dictionary, ID, size, borderBits=1)

  print('Created a marker {}.'.format(ID))
  print('Press q on the window to quit.')
  while(True):
    cv2.imshow('marker_{}'.format(ID),img)
    if cv2.waitKey(1)&0xFF==ord('q'):  break
  cv2.destroyAllWindows()

  filename= '../cpp/sample/marker/marker_{}.png'.format(ID)
  cv2.imwrite(filename, img)
  print('Saved to:',filename)

