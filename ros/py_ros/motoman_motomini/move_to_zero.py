#!/usr/bin/python
#\file    move_to_zero.py
#\brief   Move to zero pose.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.26, 2023
import roslib; roslib.load_manifest('motoman_driver')
import rospy
import actionlib
import control_msgs.msg
import trajectory_msgs.msg
import time, math, sys, copy
from get_q1 import GetState

if __name__=='__main__':
  rospy.init_node('moto_test')

  joint_names= rospy.get_param('controller_joint_names')

  client= actionlib.SimpleActionClient('/joint_trajectory_action', control_msgs.msg.FollowJointTrajectoryAction)

  # Wait some seconds for the head action server to start or exit
  if not client.wait_for_server(rospy.Duration(5.0)):
    rospy.logerr('Exiting - Joint Trajectory Action Server Not Found')
    rospy.signal_shutdown('Action Server not found')
    sys.exit(1)

  client.cancel_goal()

  goal= control_msgs.msg.FollowJointTrajectoryGoal()
  goal.trajectory.joint_names= joint_names
  def add_point(goal, time, positions, velocities):
    point= trajectory_msgs.msg.JointTrajectoryPoint()
    point.positions= copy.deepcopy(positions)
    point.velocities= copy.deepcopy(velocities)
    point.time_from_start= rospy.Duration(time)
    goal.trajectory.points.append(point)

  angles= GetState().position
  print 'current angles:',angles
  add_point(goal, 0.0, angles, [0.0]*6)
  add_point(goal, 3.0, [0.0]*6, [0.0]*6)

  goal.trajectory.header.stamp= rospy.Time.now()
  client.send_goal(goal)
  #rospy.sleep(1.0)
  #client.cancel_goal()
  #client.wait_for_result(timeout=rospy.Duration(20.0))

  print client.get_result()

  rospy.signal_shutdown('Done.')
