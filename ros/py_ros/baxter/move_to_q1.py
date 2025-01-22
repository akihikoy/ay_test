#!/usr/bin/python3
#\file    move_to_q1.py
#\brief   Baxter: move to a joint angle vector
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.08, 2015

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

  # Set joint position speed ratio for execution.
  # joint position motion speed ratio [0.0-1.0].
  limbs[RIGHT].set_joint_position_speed(0.3)
  limbs[LEFT].set_joint_position_speed(0.3)

  #NOTE: set_joint_positions seems working only when the target position is close to the current position.
  #  while move_to_joint_positions seems working from any positions.
  #  move_to_joint_positions(angles, timeout=20.0, threshold=0.01)
  #  threshold: joint position accuracy (rad) at which waypoints must achieve;
  #    default is defined as settings.JOINT_ANGLE_TOLERANCE, which is 0.008726646

  print(joint_names[RIGHT])
  print([limbs[RIGHT].joint_angle(joint) for joint in joint_names[RIGHT]])
  q0=[ 0.70, 0.02,  0.05, 1.51,  1.05, 0.18, -0.41]
  q1=[-0.70, 0.02, -0.05, 1.51, -1.05, 0.18,  0.41]
  angles= {joint:q0[j] for j,joint in enumerate(joint_names[RIGHT])}  #Deserialize
  print(angles)
  #limbs[RIGHT].set_joint_positions(angles)
  #time.sleep(1.0)
  limbs[RIGHT].move_to_joint_positions(angles, timeout=20.0, threshold=0.01)
  angles= {joint:q1[j] for j,joint in enumerate(joint_names[LEFT])}  #Deserialize
  #limbs[LEFT].set_joint_positions(angles)
  #time.sleep(1.0)
  limbs[LEFT].move_to_joint_positions(angles, timeout=20.0, threshold=0.01)
  #print angles
  #print {joint_names[RIGHT][0]:limbs[RIGHT].joint_angle(joint_names[RIGHT][0])-0.5}
  #limbs[RIGHT].set_joint_positions({joint_names[RIGHT][0]:limbs[RIGHT].joint_angle(joint_names[RIGHT][0])-0.5})
  #limbs[LEFT].set_joint_positions({joint_names[LEFT][0]:limbs[LEFT].joint_angle(joint_names[LEFT][0])-0.5})
  #angles= {joint_names[RIGHT][0]:q0[0], joint_names[RIGHT][1]:q0[1], joint_names[RIGHT][2]:q0[2], joint_names[RIGHT][3]:q0[3], joint_names[RIGHT][4]:q0[4], joint_names[RIGHT][5]:q0[5], joint_names[RIGHT][6]:q0[6]}
  #print angles
  #limbs[RIGHT].set_joint_positions(angles)
  #time.sleep(1.0)

  rospy.signal_shutdown('Done.')
