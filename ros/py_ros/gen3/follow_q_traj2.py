#!/usr/bin/python
#\file    follow_q_traj1.py
#\brief   Follow a joint angle trajectory.
#         WARNING: This code does not work as of Nov 26 2019
#         since PreComputedJointTrajectory (used in implementing follow_joint_trajectory)
#         rejects trajectories that do not
#         have 1msec timesteps intervals between all trajectory points.
#         cf.
#         https://github.com/Kinovarobotics/matlab_kortex/blob/master/simplified_api/documentation/precomputed_joint_trajectories.md#hard-limits-and-conditions-to-respect
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.25, 2019

import roslib
import rospy
import actionlib
import control_msgs.msg
import trajectory_msgs.msg
import sensor_msgs.msg
import sys, copy

if __name__=='__main__':
  rospy.init_node('gen3_test')

  joint_names= ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

  client= actionlib.SimpleActionClient('/gen3a/gen3_joint_trajectory_controller/follow_joint_trajectory', control_msgs.msg.FollowJointTrajectoryAction)
  client.cancel_goal()  #Ensure to cancel the ongoing goal.

  # Wait some seconds for the head action server to start or exit
  if not client.wait_for_server(rospy.Duration(5.0)):
    rospy.logerr('Exiting - Joint Trajectory Action Server Not Found')
    rospy.signal_shutdown('Action Server not found')
    sys.exit(1)

  goal= control_msgs.msg.FollowJointTrajectoryGoal()
  #goal.goal_time_tolerance= rospy.Time(0.1)
  goal.trajectory.joint_names= joint_names
  def add_point(goal, time, positions, velocities):
    point= trajectory_msgs.msg.JointTrajectoryPoint()
    point.positions= copy.deepcopy(positions)
    point.velocities= copy.deepcopy(velocities)
    point.time_from_start= rospy.Duration(time)
    goal.trajectory.points.append(point)

  angles= rospy.wait_for_message('/gen3a/joint_states', sensor_msgs.msg.JointState, 5.0).position
  add_point(goal, 0.0, angles, [0.0]*6)
  add_point(goal, 1.0, [q+0.02 for q in angles], [0.0]*6)
  add_point(goal, 3.0, [q-0.02 for q in angles], [0.0]*6)
  add_point(goal, 4.0, angles, [0.0]*6)

  goal.trajectory.header.stamp= rospy.Time.now()
  client.send_goal(goal)
  #client.cancel_goal()
  #client.wait_for_result(timeout=rospy.Duration(20.0))

  print client.get_result()

  #rospy.signal_shutdown('Done.')
