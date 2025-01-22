#!/usr/bin/python3
#\file    move_to_q2.py
#\brief   Baxter: move to a joint angle vector
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.16, 2016
#cf. move_to_q1.py
#HOW TO MAKE SHAKING MOTION?
#cf. follow_q_traj2.py

import roslib
import rospy
import baxter_interface
import time, random

RIGHT=0
LEFT=1
def LRTostr(whicharm):
  if whicharm==RIGHT: return 'right'
  if whicharm==LEFT:  return 'left'
  return None

#Generate a random number of uniform distribution of specified bound.
def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

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

  # Set joint position speed ratio for execution.
  # joint position motion speed ratio [0.0-1.0].
  limbs[RIGHT].set_joint_position_speed(1.0)
  for i in range(10):
    q0=[ 0.70, 0.02,  0.05, 1.51,  1.05, 0.18, -0.41]
    #q0=[q+Rand(-0.1,0.1) for q in q0]
    q0=[q+(0.1 if i%2==0 else -0.1) for q in q0]
    angles= {joint:q0[j] for j,joint in enumerate(joint_names[RIGHT])}  #Deserialize
    limbs[RIGHT].move_to_joint_positions(angles, timeout=0.3, threshold=0.01)
    #limbs[RIGHT].set_joint_positions(angles,raw=True)

  q0=[ 0.70, 0.02,  0.05, 1.51,  1.05, 0.18, -0.41]
  angles= {joint:q0[j] for j,joint in enumerate(joint_names[RIGHT])}  #Deserialize
  limbs[RIGHT].move_to_joint_positions(angles, timeout=0.3, threshold=0.01)

  rospy.signal_shutdown('Done.')
