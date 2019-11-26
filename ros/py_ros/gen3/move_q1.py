#!/usr/bin/python
#\file    move_q1.py
#\brief   Move the robot to a target joint angles.
#         It increments all joint angles in 0.02 rad.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.25, 2019
#ref. ros_kortex/kortex_examples/python/full_arm/example_full_arm_movement.py
import roslib
import rospy
import sensor_msgs.msg
import kortex_driver.msg
import kortex_driver.srv
import math
import numpy as np

# Matlab-like mod function that returns always positive
def Mod(x, y):
  if y==0:  return x
  return x-y*math.floor(x/y)
#Convert radian to [0,2*pi)
def AngleMod2(q):
  return Mod(q,math.pi*2.0)

if __name__=='__main__':
  rospy.init_node('gen3_test')
  rospy.wait_for_service('/gen3a/base/play_joint_trajectory')
  srv_play_joint= rospy.ServiceProxy('/gen3a/base/play_joint_trajectory', kortex_driver.srv.PlayJointTrajectory)

  q0= rospy.wait_for_message('/gen3a/joint_states', sensor_msgs.msg.JointState, 5.0).position

  req= kortex_driver.srv.PlayJointTrajectoryRequest()
  #NOTE: Joint angle must be in degrees.  It should be positive.
  #https://github.com/Kinovarobotics/kortex/blob/master/api_python/doc/markdown/messages/Base/JointAngle.md
  rad2deg= lambda q:q/np.pi*180.0
  for i in range(len(q0)):
    temp_angle= kortex_driver.msg.JointAngle()
    temp_angle.joint_identifier= i
    temp_angle.value= rad2deg(AngleMod2(q0[i]+0.02))
    req.input.joint_angles.joint_angles.append(temp_angle)

  try:
    srv_play_joint(req)
  except rospy.ServiceException:
    rospy.logerr("Failed to call play_joint_trajectory")
