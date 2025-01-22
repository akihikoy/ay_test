#!/usr/bin/python3
#\file    record_by_nav.py
#\brief   Baxter: record poses by pressing navigation button
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.29, 2015

import roslib
import rospy
import baxter_interface
import time

RIGHT=0
LEFT=1
def LRTostr(whicharm):
  if whicharm==RIGHT: return 'right'
  if whicharm==LEFT:  return 'left'
  return None

if __name__=='__main__':
  rospy.init_node('baxter_test')
  arm= RIGHT

  rs= baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
  init_state= rs.state().enabled
  def clean_shutdown():
    if not init_state:
      print('Disabling robot...')
      rs.disable()
  rospy.on_shutdown(clean_shutdown)
  rs.enable()

  limbs= [None,None]
  limbs[RIGHT]= baxter_interface.Limb(LRTostr(RIGHT))
  limbs[LEFT]=  baxter_interface.Limb(LRTostr(LEFT))

  joint_names= [[],[]]
  #joint_names[RIGHT]= ['right_'+joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
  #joint_names[LEFT]=  ['left_' +joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
  joint_names[RIGHT]= limbs[RIGHT].joint_names()
  joint_names[LEFT]=  limbs[LEFT].joint_names()

  navigator_io= [None,None]
  navigator_io[RIGHT]= baxter_interface.Navigator(LRTostr(RIGHT))
  navigator_io[LEFT]= baxter_interface.Navigator(LRTostr(LEFT))

  def print_angles(value):
    if value:
      angles_r= limbs[RIGHT].joint_angles()
      q_r= [angles_r[joint] for joint in joint_names[RIGHT]]  #Serialize
      angles_l= limbs[LEFT].joint_angles()
      q_l= [angles_l[joint] for joint in joint_names[LEFT]]  #Serialize
      print('[q_r,q_l]=',[q_r,q_l])

  # Navigator scroll wheel button press
  navigator_io[RIGHT].button0_changed.connect(print_angles)
  navigator_io[LEFT].button0_changed.connect(print_angles)
  rospy.spin()

  rospy.signal_shutdown('Done.')
