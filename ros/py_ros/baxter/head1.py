#!/usr/bin/python
#\file    head1.py
#\brief   Baxter: head control
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.08, 2015
'''
NOTE: run beforehand:
  $ rosrun baxter_interface head_action_server.py
'''

import roslib
import rospy
import actionlib
import control_msgs.msg
import baxter_interface
import time, math, sys

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

  client= actionlib.SimpleActionClient('/robot/head/head_action', control_msgs.msg.SingleJointPositionAction)

  # Wait some seconds for the head action server to start or exit
  if not client.wait_for_server(rospy.Duration(5.0)):
    rospy.logerr('Exiting - Head Action Server Not Found')
    rospy.logerr('Run: rosrun baxter_interface head_action_server.py')
    rospy.signal_shutdown('Action Server not found')
    sys.exit(1)

  def command(position, velocity):
    goal= control_msgs.msg.SingleJointPositionGoal()
    goal.position= position
    goal.max_velocity= velocity
    client.send_goal(goal)
    client.wait_for_result(timeout=rospy.Duration(10.0))
    return client.get_result()

  command(position=0.0, velocity=100.0)
  command(position=-1.57, velocity=10.0)
  command(position=0.0, velocity=80.0)
  command(position=1.0, velocity=20.0)
  command(position=0.0, velocity=60.0)

  rospy.signal_shutdown('Done.')
