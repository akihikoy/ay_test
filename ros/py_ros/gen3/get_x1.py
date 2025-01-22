#!/usr/bin/python3
#\file    get_x1.py
#\brief   Get Cartesian pose with KDL.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.2
#\date    Nov.25, 2019

import roslib
import rospy
import sensor_msgs.msg

import numpy as np
from kdl_kin2 import TKinematics

if __name__=='__main__':
  np.set_printoptions(precision=4)

  rospy.init_node('gen3_test')

  print('Using TKinematics (robot_description == Gen3 is assumed).')
  kin= TKinematics(base_link='base_link',end_link='end_effector_link',description='/gen3a/robot_description')

  q0= rospy.wait_for_message('/gen3a/joint_states', sensor_msgs.msg.JointState, 5.0).position
  angles= {joint:q0[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
  x0= kin.forward_position_kinematics(angles)
  print('q0=',np.array(q0))
  print('x0= FK(q0)=',x0)
