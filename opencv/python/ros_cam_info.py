#!/usr/bin/python
#\file    ros_cam_info.py
#\brief   Get camera parameters from a ROS topic.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.7, 2021

import roslib; roslib.load_manifest('sensor_msgs')
import rospy
import sensor_msgs.msg
import cv2
import sys
import numpy as np

def GetCameraProjectionMatrix(cam_info_topic='/camera/aligned_depth_to_color/camera_info'):
  try:
    cam_info= rospy.wait_for_message(cam_info_topic, sensor_msgs.msg.CameraInfo, 5.0)
    proj_mat= np.array(cam_info.P).reshape(3,4) #get camera projection matrix from ros topic
    return proj_mat
  except (rospy.ROSException, rospy.ROSInterruptException):
    raise Exception('Failed to read topic: {cam_info_topic}'.format(cam_info_topic=cam_info_topic))

def GetCameraInfo(cam_info_topic='/camera/aligned_depth_to_color/camera_info'):
  try:
    cam_info= rospy.wait_for_message(cam_info_topic, sensor_msgs.msg.CameraInfo, 5.0)
    P= np.array(cam_info.P).reshape(3,4)
    K= np.array(cam_info.K).reshape(3,3)
    D= np.array(cam_info.D)
    R= np.array(cam_info.R).reshape(3,3)
    return P,K,D,R
  except (rospy.ROSException, rospy.ROSInterruptException):
    raise Exception('Failed to read topic: {cam_info_topic}'.format(cam_info_topic=cam_info_topic))

if __name__=='__main__':
  rospy.init_node('ros_cam_info')
  topic= sys.argv[1] if len(sys.argv)>1 else '/camera/aligned_depth_to_color/camera_info'
  proj_mat= GetCameraProjectionMatrix(cam_info_topic=topic)
  print 'proj_mat=\n',proj_mat

  P,K,D,R= GetCameraInfo(cam_info_topic=topic)
  print 'P=\n',P
  print 'K=\n',K
  print 'D=\n',D
  print 'R=\n',R
