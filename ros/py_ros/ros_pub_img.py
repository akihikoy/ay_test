#!/usr/bin/python3
#\file    ros_pub_img.py
#\brief   Capture images from a camera and publish them as ROS image topics.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.27, 2021
'''
$ roscore
$ ./ros_pub_img.py
$ ./ros_sub_img.py
$ rosrun image_view image_view image:=/camera/color/image_raw
'''

import roslib; roslib.load_manifest('sensor_msgs')
import rospy
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
import cv2

if __name__=='__main__':
  rospy.init_node('ros_pub_img')
  pub= rospy.Publisher('/camera/color/image_raw', sensor_msgs.msg.Image, queue_size=1)

  cap= cv2.VideoCapture(0)

  while(True):
    ret,frame= cap.read()

    msg= CvBridge().cv2_to_imgmsg(frame, encoding="bgr8")
    pub.publish(msg)

    cv2.imshow('camera',frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
