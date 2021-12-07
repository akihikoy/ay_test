#!/usr/bin/python
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

def ImageCallback(msg, fmt):
  img= CvBridge().imgmsg_to_cv2(msg, fmt)
  if fmt=='16UC1':
    img_viz= cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
  else:  img_viz= img
  cv2.imshow('image', img_viz)
  key= cv2.waitKey(1)&0xFF
  if key==ord(' '):
    for i in range(10000):
      filename= '/tmp/img{:05d}.png'.format(i)
      if not os.path.exists(filename):  break
    cv2.imwrite(filename, img_viz)
    print 'Saved image into:', filename
  elif key==ord('q'):
    rospy.signal_shutdown('quit.')
    cv2.destroyAllWindows()

if __name__=='__main__':
  topic= sys.argv[1] if len(sys.argv)>1 else '/camera/color/image_raw'
  fmt= sys.argv[2] if len(sys.argv)>2 else None
  if fmt is None:
    fmt= '16UC1' if 'depth' in topic else 'bgr8'
  rospy.init_node('ros_img_save')
  image_sub= rospy.Subscriber(topic, sensor_msgs.msg.Image, lambda msg:ImageCallback(msg,fmt))
  rospy.spin()
