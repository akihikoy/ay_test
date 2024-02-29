#!/usr/bin/python
#\file    aruco_board1.py
#\brief   ArUco board detection test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.29, 2024
import cv2
import numpy as np

cap= cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

'''
cv::aruco::Board board = createBoard();

...
vector< int > markerIds;
vector< vector<Point2f> > markerCorners;
cv::aruco::detectMarkers(inputImage, board.dictionary, markerCorners, markerIds);
// if at least one marker detected
if(markerIds.size() > 0) {
    cv::Vec3d rvec, tvec;
    int valid = cv::aruco::estimatePoseBoard(markerCorners, markerIds, board, cameraMatrix, distCoeffs, rvec, tvec);
}
'''

if __name__=='__main__':
  dictionary= cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
  parameters= cv2.aruco.DetectorParameters_create()

  #NOTE: Use ../cpp/sample/marker/markers_1_6.svg as the board.

  board= cv2.aruco.GridBoard_create(markersX=2, markersY=3, markerLength=0.04, markerSeparation=0.04, dictionary=dictionary, firstMarker=1)

  #NOTE: This part should be the same as GridBoard_create but Board_create fails with:
  #  error: (-215) objPoints.type() == CV_32FC3
  #markersX, markersY, markerLength, markerSeparation= 2, 3, 0.04, 0.04
  #objPoints, ids= [], []
  #i= 1
  #for r in range(markersY):
    #for c in range(markersX):
      #pts= np.array([[-c*(markerLength+markerSeparation), r*(markerLength+markerSeparation), 0.0],
                    #[-(c*(markerLength+markerSeparation)+markerLength), r*(markerLength+markerSeparation), 0.0],
                    #[-(c*(markerLength+markerSeparation)+markerLength), r*(markerLength+markerSeparation)+markerLength, 0.0],
                    #[-c*(markerLength+markerSeparation), r*(markerLength+markerSeparation)+markerLength, 0.0],
                    #], dtype=np.float32)
      #objPoints.append(pts)
      #ids.append([i])
      #i+= 1
  #board= cv2.aruco.Board_create(np.array(objPoints).astype(np.float32), dictionary, np.array(ids))

  print 'board=', board

  Alpha= 1.0
  K= np.array([ 2.7276124573617790e+02, 0., 3.2933938280614751e+02, 0.,
              2.7301706435392521e+02, 2.4502942171375494e+02, 0., 0., 1. ]).reshape(3,3)
  D= np.array([ -2.7764926296767156e-01, 7.9998204984529891e-02,
          -8.3937489998474823e-04, 6.0999999999999943e-04,
          -7.4899999999988864e-03 ])
  size_in,size_out=  (640,480),(640,480)
  P,_= cv2.getOptimalNewCameraMatrix(K, D, size_in, Alpha, size_out)
  print P, P.shape, P.dtype

  while(True):
    ret,frame= cap.read()

    corners, ids, rejectedImgPoints= cv2.aruco.detectMarkers(frame, board.dictionary, parameters=parameters)
    if ids is not None and len(ids)>0:
      cv2.aruco.drawDetectedMarkers(frame, corners, ids)
      #print 'corners:', corners
      retval, rvec, tvec= cv2.aruco.estimatePoseBoard(corners, ids, board, P, D)
      print 'retval, rvec, tvec=', retval, rvec, tvec
      #draw the axis
      #cv2.drawFrameAxes(frame, P, D, rvec, tvec, length=0.05)  #For OpenCV 3.4+
      cv2.aruco.drawAxis(frame, P, D, rvec, tvec, length=0.05);

    cv2.imshow('marker_detection',frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


