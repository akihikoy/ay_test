#!/usr/bin/python3
#\file    ros_img_save.py
#\brief   Save a ROS image topic.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.7, 2021

import roslib; roslib.load_manifest('sensor_msgs')
import rospy
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys, os
import numpy as np

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
  cv2.imshow('image', img_viz)
  key= cv2.waitKey(1)&0xFF
  if key==ord(' '):
    for i in range(10000):
      filename= '/tmp/img{:05d}({}).png'.format(i,fmt)
      if not os.path.exists(filename):  break
    #cv2.imwrite(filename, img_viz)
    cv2.imwrite(filename, img)
    print('Saved image into: {} ({})'.format(filename,fmt))
    #NOTE: Use cv2.imread(filename, cv2.IMREAD_ANYDEPTH) to read depth image (16UC1).
  elif key==ord('q'):
    rospy.signal_shutdown('quit.')
    cv2.destroyAllWindows()

if __name__=='__main__':
  topic= sys.argv[1] if len(sys.argv)>1 else '/camera/color/image_raw'
  fmt= sys.argv[2] if len(sys.argv)>2 else None
  rospy.init_node('ros_img_save')
  if fmt is None:
    #fmt= '16UC1' if 'depth' in topic else 'bgr8'
    fmt= GetImageEncoding(topic, convert_cv=True)
  image_sub= rospy.Subscriber(topic, sensor_msgs.msg.Image, lambda msg:ImageCallback(msg,fmt))
  rospy.spin()
