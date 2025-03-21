#!/usr/bin/python3
#\file    follow_q_traj1.py
#\brief   Follow a joint angle trajectory.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.12, 2018
#Based on: ../baxter/follow_q_traj1.py

import roslib
import rospy
import actionlib
import control_msgs.msg
import trajectory_msgs.msg
import time, math, sys, copy
from get_q1 import GetState

if __name__=='__main__':
  rospy.init_node('ur_test')

  joint_names= ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

  client= actionlib.SimpleActionClient('/follow_joint_trajectory', control_msgs.msg.FollowJointTrajectoryAction)

  # Wait some seconds for the head action server to start or exit
  if not client.wait_for_server(rospy.Duration(5.0)):
    rospy.logerr('Exiting - Joint Trajectory Action Server Not Found')
    rospy.signal_shutdown('Action Server not found')
    sys.exit(1)

  goal= control_msgs.msg.FollowJointTrajectoryGoal()
  #goal.goal_time_tolerance= rospy.Time(0.1)
  goal.trajectory.joint_names= joint_names
  #NOTE: We need to specify velocities. Otherwise:
  #error_code: -1
  #error_string: "Received a goal without velocities"
  def add_point(goal, time, positions, velocities):
    point= trajectory_msgs.msg.JointTrajectoryPoint()
    point.positions= copy.deepcopy(positions)
    point.velocities= copy.deepcopy(velocities)
    point.time_from_start= rospy.Duration(time)
    goal.trajectory.points.append(point)

  angles= GetState().position
  add_point(goal, 0.0, angles, [0.0]*6)
  add_point(goal, 2.0, [2.2,0,-1.57,0,0,0], [0.0]*6)
  add_point(goal, 4.0, [1.5,0,-1.57,0,0,0], [0.0]*6)
  add_point(goal, 6.0, [1.5,-0.2,-1.57,0,0,0], [0.0]*6)

  goal.trajectory.header.stamp= rospy.Time.now()
  client.send_goal(goal)
  #client.cancel_goal()
  client.wait_for_result(timeout=rospy.Duration(20.0))

  print(client.get_result())

  rospy.signal_shutdown('Done.')
