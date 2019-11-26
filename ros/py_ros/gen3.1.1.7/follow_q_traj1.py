#!/usr/bin/python
#\file    follow_q_traj1.py
#\brief   Follow a joint angle trajectory.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.10, 2019
import roslib
import rospy
import math
import sensor_msgs.msg
import kortex_driver.srv
import kortex_driver.msg

if __name__=='__main__':
  rospy.init_node('gen3_test')

  rospy.wait_for_service('PlayJointTrajectory')
  srvPlayJointTrajectory= rospy.ServiceProxy('PlayJointTrajectory', kortex_driver.srv.PlayJointTrajectory)

  q0= rospy.wait_for_message('/joint_states', sensor_msgs.msg.JointState, 5.0).position

  q_trg= [q+0.02 for q in q0]

  req= kortex_driver.srv.PlayJointTrajectoryRequest()
  req.input= kortex_driver.msg.ConstrainedJointAngles()
  req.input.joint_angles= kortex_driver.msg.JointAngles()
  req.input.joint_angles.joint_angles= [kortex_driver.msg.JointAngle(0,q/math.pi*180.0) for q in q_trg]

  req.input.constraint= kortex_driver.msg.JointTrajectoryConstraint()
  req.input.constraint.type= kortex_driver.msg.JointTrajectoryConstraintType.JOINT_CONSTRAINT_DURATION
  req.input.constraint.value= 3.0

  srvPlayJointTrajectory(req)

