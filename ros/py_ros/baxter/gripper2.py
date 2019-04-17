#!/usr/bin/python
#\file    gripper2.py
#\brief   We investigate the relation between command (0-100) and position (meters).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.07, 2015

import roslib
import rospy
import baxter_interface
import time, math

RIGHT=0
LEFT=1
def LRTostr(whicharm):
  if whicharm==RIGHT: return 'right'
  if whicharm==LEFT:  return 'left'
  return None

if __name__=='__main__':
  rospy.init_node('baxter_test')
  #arm= LEFT
  #limb= baxter_interface.Limb(LRTostr(arm))

  rs= baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
  init_state= rs.state().enabled
  def clean_shutdown():
    if not init_state:
      print 'Disabling robot...'
      rs.disable()
  rospy.on_shutdown(clean_shutdown)
  rs.enable()

  epgripper= baxter_interface.Gripper('right', baxter_interface.CHECK_VERSION)

  #Calibration
  print 'Calibrating electric parallel gripper...'
  epgripper.calibrate()
  print '...done'

  epgripper.set_velocity(100.0)
  epgripper.set_moving_force(100.0)
  epgripper.set_holding_force(100.0)
  #NOTE: After calibration, these parameters are set default.
  print 'gripper velocity:', epgripper.parameters()['velocity']  #Range: [0.0, 100.0], default: 50.0
  print 'gripper dead_zone:', epgripper.parameters()['dead_zone']  #Range: ???, default: 5.0
  print 'gripper moving_force:', epgripper.parameters()['moving_force']  #Range: [0.0, 100.0], default: 40.0
  print 'gripper holding_force:', epgripper.parameters()['holding_force']  #Range: [0.0, 100.0], default: 30.0

  while True:
    c= raw_input('Type 0-100 or {o,c,q} > ')
    if c=='q':  break
    elif c=='o':  epgripper.open(block=True)
    elif c=='c':  epgripper.close(block=True)
    else:
      cmd= int(c)
      epgripper.command_position(cmd,block=True)
    print 'Position: %r'%(epgripper.position())

  rospy.signal_shutdown('Done.')

