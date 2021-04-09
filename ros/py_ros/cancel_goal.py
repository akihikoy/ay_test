#!/usr/bin/python
#\file    cancel_goal.py
#\brief   Cancel the trajectory control.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.09, 2021

import roslib
import rospy
import actionlib
import sensor_msgs.msg
import control_msgs.msg
import trajectory_msgs.msg
import copy
from kbhit2 import KBHAskGen

def GetState():
  try:
    state= rospy.wait_for_message('/joint_states', sensor_msgs.msg.JointState, 5.0)
    return state
  except (rospy.ROSException, rospy.ROSInterruptException):
    raise Exception('Failed to read topic: /joint_states')

if __name__=='__main__':
  rospy.init_node('ur_test')

  joint_names= ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

  client= actionlib.SimpleActionClient('/follow_joint_trajectory', control_msgs.msg.FollowJointTrajectoryAction)

  print 'Press space to cancel the trajectory.'
  if KBHAskGen(' ','q'):
    client.cancel_goal()
    #client.wait_for_result(timeout=rospy.Duration(20.0))

    goal= control_msgs.msg.FollowJointTrajectoryGoal()
    goal.trajectory.joint_names= joint_names
    def add_point(goal, time, positions, velocities):
      point= trajectory_msgs.msg.JointTrajectoryPoint()
      point.positions= copy.deepcopy(positions)
      point.velocities= copy.deepcopy(velocities)
      point.time_from_start= rospy.Duration(time)
      goal.trajectory.points.append(point)
    angles= GetState().position
    add_point(goal, 0.0, angles, [0.0]*6)
    add_point(goal, 0.1, angles, [0.0]*6)
    goal.trajectory.header.stamp= rospy.Time.now()
    client.send_goal(goal)
    client.wait_for_result(timeout=rospy.Duration(20.0))
