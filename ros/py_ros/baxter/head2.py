#!/usr/bin/python
#\file    head2.py
#\brief   Baxter: head control 2
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.09, 2015

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

  head= baxter_interface.Head()

  head.set_pan(0.0)  #NOTE: Default speed=100, timeout=10
  head.command_nod()
  head.set_pan(0.0)
  print 'Head pan=',head.pan()
  head.set_pan(-1.57, speed=10, timeout=10)  #NOTE: Set timeout=0 for non-blocking
  print 'Head pan=',head.pan()
  head.set_pan(0.0, speed=80)
  print 'Head pan=',head.pan()
  head.set_pan(1.0, speed=20)
  print 'Head pan=',head.pan()
  head.set_pan(0.0, speed=10, timeout=0)
  head.command_nod()
  print 'Head pan=',head.pan()
  head.set_pan(0.0)
  head.command_nod()

  rospy.signal_shutdown('Done.')

