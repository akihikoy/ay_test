#!/usr/bin/python3
#\file    follow_q_traj_stop1.py
#\brief   Follow a trajectory and stop when indicated by the operator.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.26, 2023
import roslib; roslib.load_manifest('motoman_driver')
import rospy
import actionlib
import control_msgs.msg
import trajectory_msgs.msg
import actionlib_msgs.msg
import sensor_msgs.msg
import time, math, sys, copy
#from get_q1 import GetState
import get_q2
from kbhit2 import TKBHit

StateToStr= {getattr(actionlib_msgs.msg.GoalStatus,key):key for key in ('PENDING', 'ACTIVE', 'RECALLED', 'REJECTED', 'PREEMPTED', 'ABORTED', 'SUCCEEDED', 'LOST')}
print('StateToStr=',StateToStr)

if __name__=='__main__':
  rospy.init_node('moto_test')

  joint_names= rospy.get_param('controller_joint_names')
  dof= len(joint_names)

  sub_js= rospy.Subscriber('/joint_states', sensor_msgs.msg.JointState, get_q2.JointStatesCallback)

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

  for i in range(100):
    if get_q2.q_curr is not None:  break
    print(get_q2.q_curr)
    rospy.sleep(0.1)
  #angles= GetState().position
  angles= get_q2.q_curr
  print('current angles:',angles)
  dt= 1.0
  add_point(goal, 0.0, angles, [0.0]*dof)
  add_point(goal, dt*1.0, [0.6, -0.6, -0.1, 0.4], [0.0]*dof)
  add_point(goal, dt*3.0, [-0.6, 0.6, -0.15, -0.4], [0.0]*dof)
  add_point(goal, dt*4.0, angles, [0.0]*dof)

  with TKBHit() as kbhit:
    rate_adjustor= rospy.Rate(50)
    goal.trajectory.header.stamp= rospy.Time.now()
    client.send_goal(goal)
    print('trajectory started. hit space to stop.')
    while not rospy.is_shutdown():
      c= kbhit.KBHit()
      if c==' ':
        print('stopping the trajectory...')
        #Method-1: Basically stops, but sometimes (especially in the last period) does not stop.
        #  No message from MotoROS.
        client.cancel_goal()
        #Method-2: Failed with: Joint trajectory action failed on empty trajectory.
        #goal_empty= control_msgs.msg.FollowJointTrajectoryGoal()
        #goal_empty.trajectory.joint_names= joint_names
        #goal_empty.trajectory.header.stamp= rospy.Time(0,0)
        #client.send_goal(goal_empty)
        #Method-3: Basically stops, but sometimes (especially in the last period) does not stop.
        #  Received new goal, canceling current goal
        #  Validation failed: Trajectory doesn't start at current position.
        #goal_stop= control_msgs.msg.FollowJointTrajectoryGoal()
        #goal_stop.trajectory.joint_names= joint_names
        #add_point(goal_stop, 0.0, get_q2.q_curr, [0.0]*dof)
        #goal_stop.trajectory.header.stamp= rospy.Time.now()
        #client.send_goal(goal_stop)
        break
      if client.get_state() != actionlib_msgs.msg.GoalStatus.ACTIVE:
        print('state: {} ({})'.format(StateToStr[client.get_state()], client.get_state()))
      if client.get_state() not in (actionlib_msgs.msg.GoalStatus.PENDING, actionlib_msgs.msg.GoalStatus.ACTIVE):
        break
      rate_adjustor.sleep()
  print(client.get_result())

  rospy.signal_shutdown('Done.')
