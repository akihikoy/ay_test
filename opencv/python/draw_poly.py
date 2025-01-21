#!/usr/bin/python3
#\file    draw_poly.py
#\brief   Draw polygon.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.20, 2018
import numpy as np
import cv2

cap= cv2.VideoCapture(0)

while(True):
  ret,frame= cap.read()

  #for pts in [[[213, 71], [165, 122], [159, 198], [168, 248], [202, 326], [265, 335], [318, 314], [331, 243], [354, 151], [319, 86], [248, 65]],
    #[[220, 182], [168, 184], [168, 209], [211, 219], [231, 207], [249, 187]],
    #[[256, 195], [273, 219], [316, 219], [328, 200], [322, 185], [288, 183]],
    #[[248, 216], [232, 244], [221, 254], [246, 260], [268, 254]],
    #[[245, 288], [214, 292], [246, 305], [275, 290]]]:
  pts= np.array([[100,50],[200,300],[600,200],[500,100]], np.int32)
  pts= np.array(pts, np.int32)
  pts= pts.reshape((-1,1,2))
  #cv2.fillPoly(frame, [pts], (128,0,128))
  cv2.polylines(frame, [pts], True, (255,0,128), 5)

  cv2.imshow('camera',frame)
  if cv2.waitKey(1)&0xFF==ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
