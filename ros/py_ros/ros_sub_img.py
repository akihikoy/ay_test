#!/usr/bin/python3
#\file    ros_sub_img.py
#\brief   Subscribe a ROS image topic.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.27, 2021
'''
$ roscore
$ ./ros_pub_img.py
$ ./ros_sub_img.py
'''

import roslib; roslib.load_manifest('sensor_msgs')
import rospy
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
import cv2

def ImageCallback(msg):
  img= CvBridge().imgmsg_to_cv2(msg, "bgr8")
  img= cv2.flip(img, 1)
  cv2.imshow('image', img)
  if cv2.waitKey(1)&0xFF==ord('q'):
    rospy.signal_shutdown('quit.')
    cv2.destroyAllWindows()

def OnMouse(event, x, y, flags, param):
  if event==cv2.EVENT_LBUTTONDOWN:
    print('LBUTTONDOWN',x,y)

if __name__=='__main__':
  cv2.namedWindow('image')
  cv2.setMouseCallback('image', OnMouse)
  rospy.init_node('ros_sub_img')
  image_sub= rospy.Subscriber("/camera/color/image_raw", sensor_msgs.msg.Image, ImageCallback)
  rospy.spin()
