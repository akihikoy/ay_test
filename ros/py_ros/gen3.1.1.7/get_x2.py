#!/usr/bin/python3
#\file    get_x2.py
#\brief   Get Cartesian pose with Gen3's service.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.10, 2019
import roslib
import rospy
import tf
import kortex_driver.srv
import kortex_driver.msg
import numpy as np

if __name__=='__main__':
  np.set_printoptions(precision=4)

  rospy.init_node('gen3_test')
  rospy.wait_for_service('RefreshFeedback')
  srvRefreshFeedback= rospy.ServiceProxy('RefreshFeedback', kortex_driver.srv.RefreshFeedback)

  deg2rad= lambda q:q/180.0*np.pi

  feedback= srvRefreshFeedback()
  x= feedback.output.base.tool_pose_x
  y= feedback.output.base.tool_pose_y
  z= feedback.output.base.tool_pose_z
  theta_x= deg2rad(feedback.output.base.tool_pose_theta_x)
  theta_y= deg2rad(feedback.output.base.tool_pose_theta_y)
  theta_z= deg2rad(feedback.output.base.tool_pose_theta_z)

  Q= tf.transformations.quaternion_from_euler(theta_x,theta_y,theta_z)
  print(np.array([x,y,z]+list(Q)))
