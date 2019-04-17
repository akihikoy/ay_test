#!/usr/bin/python
#\file    follow_q_traj2.py
#\brief   Baxter: follow a joint angle trajectory
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.16, 2016
#HOW TO MAKE SHAKING MOTION?
#cf. move_to_q2.py
'''
NOTE: run beforehand:
  $ rosrun baxter_interface joint_trajectory_action_server.py
'''

import roslib
import rospy
import actionlib
import control_msgs.msg
import trajectory_msgs.msg
import baxter_interface
import time, math, sys, copy

RIGHT=0
LEFT=1
def LRTostr(whicharm):
  if whicharm==RIGHT: return 'right'
  if whicharm==LEFT:  return 'left'
  return None

if __name__=='__main__':
  rospy.init_node('baxter_test')

  rs= baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
  init_state= rs.state().enabled
  def clean_shutdown():
    if not init_state:
      print 'Disabling robot...'
      rs.disable()
  rospy.on_shutdown(clean_shutdown)
  rs.enable()

  arm= RIGHT
  limbs= [None,None]
  limbs[RIGHT]= baxter_interface.Limb(LRTostr(RIGHT))
  limbs[LEFT]=  baxter_interface.Limb(LRTostr(LEFT))

  joint_names= [[],[]]
  #joint_names[RIGHT]= ['right_'+joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
  #joint_names[LEFT]=  ['left_' +joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
  joint_names[RIGHT]= limbs[RIGHT].joint_names()
  joint_names[LEFT]=  limbs[LEFT].joint_names()


  client= actionlib.SimpleActionClient('/robot/limb/%s/follow_joint_trajectory'%LRTostr(arm), control_msgs.msg.FollowJointTrajectoryAction)

  # Wait some seconds for the head action server to start or exit
  if not client.wait_for_server(rospy.Duration(5.0)):
    rospy.logerr('Exiting - Joint Trajectory Action Server Not Found')
    rospy.logerr('Run: rosrun baxter_interface joint_trajectory_action_server.py')
    rospy.signal_shutdown('Action Server not found')
    sys.exit(1)

  goal= control_msgs.msg.FollowJointTrajectoryGoal()
  goal.goal_time_tolerance= rospy.Time(0.1)
  goal.trajectory.joint_names= joint_names[arm]
  def add_point(goal, time, positions):
    point= trajectory_msgs.msg.JointTrajectoryPoint()
    point.positions= copy.deepcopy(positions)
    point.time_from_start= rospy.Duration(time)
    goal.trajectory.points.append(point)

  angles= limbs[arm].joint_angles()
  add_point(goal, 0.0, [angles[joint] for joint in joint_names[arm]])
  t0= 1.0

  q0=[[ 0.70, 0.02,  0.05, 1.51,  1.05, 0.18, -0.41],
      [-0.70, 0.02, -0.05, 1.51, -1.05, 0.18,  0.41]][arm]
  add_point(goal, t0-0.3, q0 )
  for i in range(10):
    add_point(goal, t0, [q+(0.03 if i%2==0 else -0.03) for q in q0] )
    t0+= 0.2
  add_point(goal, t0, q0 )

  goal.trajectory.header.stamp= rospy.Time.now()
  client.send_goal(goal)
  #client.cancel_goal()
  client.wait_for_result(timeout=rospy.Duration(20.0))

  print client.get_result()

  rospy.signal_shutdown('Done.')
