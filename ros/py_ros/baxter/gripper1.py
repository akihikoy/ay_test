#!/usr/bin/python
#\file    gripper1.py
#\brief   Baxter: gripper control
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.08, 2015

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
  epgripper.set_moving_force(20.0)
  epgripper.set_holding_force(20.0)
  #NOTE: After calibration, these parameters are set default.
  print 'gripper velocity:', epgripper.parameters()['velocity']  #Range: [0.0, 100.0], default: 50.0
  print 'gripper dead_zone:', epgripper.parameters()['dead_zone']  #Range: ???, default: 5.0
  print 'gripper moving_force:', epgripper.parameters()['moving_force']  #Range: [0.0, 100.0], default: 40.0
  print 'gripper holding_force:', epgripper.parameters()['holding_force']  #Range: [0.0, 100.0], default: 30.0

  print 'Close, open, ...'
  for i in xrange(2):
    epgripper.close()
    time.sleep(1)
    print 'current: %r'%(epgripper.position())
    epgripper.open()
    time.sleep(1)
    print 'current: %r'%(epgripper.position())

  ti= 0.0
  while ti<1.0:
    goal= 100.0*(0.5+0.5*math.cos(math.pi*ti))
    epgripper.command_position(goal)
    time.sleep(0.2)
    print '@%f goal/current: %r / %r'%(ti, goal, epgripper.position())
    ti+= 0.02

  rospy.signal_shutdown('Done.')

