#!/usr/bin/python3
#\file    move_x1.py
#\brief   Moving the robot in Cartesian space.
#         The robot moves 5 cm above of the current pose.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.25, 2019
#ref. ros_kortex/kortex_examples/python/full_arm/example_full_arm_movement.py
import roslib
import rospy
import tf
import kortex_driver.msg
import kortex_driver.srv
import numpy as np

if __name__=='__main__':
  rospy.init_node('gen3_test')
  rospy.wait_for_service('/gen3a/base/play_cartesian_trajectory')
  srv_play_cart= rospy.ServiceProxy('/gen3a/base/play_cartesian_trajectory', kortex_driver.srv.PlayCartesianTrajectory)

  feedback= rospy.wait_for_message('/gen3a/base_feedback', kortex_driver.msg.BaseCyclic_Feedback)

  req= kortex_driver.srv.PlayCartesianTrajectoryRequest()
  req.input.target_pose.x= feedback.base.commanded_tool_pose_x
  req.input.target_pose.y= feedback.base.commanded_tool_pose_y
  req.input.target_pose.z= feedback.base.commanded_tool_pose_z + 0.05
  req.input.target_pose.theta_x= feedback.base.commanded_tool_pose_theta_x
  req.input.target_pose.theta_y= feedback.base.commanded_tool_pose_theta_y
  req.input.target_pose.theta_z= feedback.base.commanded_tool_pose_theta_z

  pose_speed= kortex_driver.msg.CartesianSpeed()
  pose_speed.translation= 0.01
  pose_speed.orientation= 1.5

  # The constraint is a one_of in Protobuf. The one_of concept does not exist in ROS
  # To specify a one_of, create it and put it in the appropriate list of the oneof_type member of the ROS object :
  req.input.constraint.oneof_type.speed.append(pose_speed)

  # Call the service
  try:
    srv_play_cart(req)
  except rospy.ServiceException:
    rospy.logerr('Failed to call play_cartesian_trajectory')
