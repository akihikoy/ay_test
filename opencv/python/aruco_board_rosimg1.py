#!/usr/bin/python
#\file    aruco_board_rosimg1.py
#\brief   ArUco board detection on ROS image topic.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.29, 2024
import cv2
import numpy as np
import roslib; roslib.load_manifest('sensor_msgs')
import rospy
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
import sys, os

dictionary= cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters= cv2.aruco.DetectorParameters_create()

#NOTE: Use ../cpp/sample/marker/markers_1_9.svg as the board.
board= cv2.aruco.GridBoard_create(markersX=3, markersY=3, markerLength=0.04, markerSeparation=0.02, dictionary=dictionary, firstMarker=1)

print 'board=', board

rospy.init_node('aruco_board_rosimg1')

from ros_cam_info import *
P,K,D,R= GetCameraInfo()
P= P[:3,:3]
Alpha= 1.0


#Get the image encoding of a ROS image topic.
#If convert_cv is true, the encoding is converted for OpenCV image conversion.
def GetImageEncoding(img_topic, convert_cv=False, time_out=5.0):
  try:
    msg= rospy.wait_for_message(img_topic, sensor_msgs.msg.Image, time_out)
    encoding= msg.encoding
    if not convert_cv:  return encoding
    if encoding=="rgb8":  return "bgr8"
    if encoding=="RGB8":  return "BGR8"
    #TODO: Add more conversion if necessary.
    return encoding;
  except (rospy.ROSException, rospy.ROSInterruptException):
    raise Exception('Failed to receive the image topic: {}'.format(img_topic))

def ImageCallback(msg, fmt):
  img= CvBridge().imgmsg_to_cv2(msg, fmt)
  if fmt=='16UC1':
    img_viz= cv2.cvtColor((img).astype('uint8'), cv2.COLOR_GRAY2BGR)
    #print np.min(img),np.max(img),'-->',np.min(img_viz),np.max(img_viz)
  else:  img_viz= img
  
  frame= img_viz
  print 'frame=',frame.shape
  print 'dtype=',frame.dtype

  corners, ids, rejectedImgPoints= cv2.aruco.detectMarkers(frame, board.dictionary, parameters=parameters)
  if ids is not None and len(ids)>0:
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    #print 'corners:', corners
    retval, rvec, tvec= cv2.aruco.estimatePoseBoard(corners, ids, board, P, D)
    print 'retval=', retval
    print 'rvec=', rvec
    print 'tvec=', tvec
    #draw the axis
    #cv2.drawFrameAxes(frame, P, D, rvec, tvec, length=0.05)  #For OpenCV 3.4+
    cv2.aruco.drawAxis(frame, P, D, rvec, tvec, length=0.05);

  cv2.imshow('marker_detection',frame)

  key= cv2.waitKey(1)&0xFF
  if key==ord('q'):
    rospy.signal_shutdown('quit.')
    cv2.destroyAllWindows()

if __name__=='__main__':
  topic= sys.argv[1] if len(sys.argv)>1 else '/camera/color/image_raw'
  fmt= sys.argv[2] if len(sys.argv)>2 else None
  if fmt is None:
    #fmt= '16UC1' if 'depth' in topic else 'bgr8'
    fmt= GetImageEncoding(topic, convert_cv=True)
  image_sub= rospy.Subscriber(topic, sensor_msgs.msg.Image, lambda msg:ImageCallback(msg,fmt))
  rospy.spin()


