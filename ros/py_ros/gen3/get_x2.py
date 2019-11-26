#!/usr/bin/python
#\file    get_x2.py
#\brief   Get Cartesian pose from Gen3's topic.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.25, 2019
import roslib
import rospy
import tf
import kortex_driver.msg
import numpy as np

if __name__=='__main__':
  np.set_printoptions(precision=4)

  rospy.init_node('gen3_test')

  feedback= rospy.wait_for_message('/gen3a/base_feedback', kortex_driver.msg.BaseCyclic_Feedback, 5.0)

  deg2rad= lambda q:q/180.0*np.pi

  x= feedback.base.tool_pose_x
  y= feedback.base.tool_pose_y
  z= feedback.base.tool_pose_z
  theta_x= deg2rad(feedback.base.tool_pose_theta_x)
  theta_y= deg2rad(feedback.base.tool_pose_theta_y)
  theta_z= deg2rad(feedback.base.tool_pose_theta_z)

  Q= tf.transformations.quaternion_from_euler(theta_x,theta_y,theta_z)
  print np.array([x,y,z]+list(Q))
