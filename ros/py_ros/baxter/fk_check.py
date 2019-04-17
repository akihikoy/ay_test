#!/usr/bin/python
#\file    fk_check.py
#\brief   Baxter: check the FK correctness.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.09, 2015

import roslib
import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics
import time
from rviz1 import TSimpleVisualizer

RIGHT=0
LEFT=1
def LRTostr(whicharm):
  if whicharm==RIGHT: return 'right'
  if whicharm==LEFT:  return 'left'
  return None

#Convert Baxter pose to x
def BPoseToX(pose):
  x= [0]*7
  position,orientation= pose['position'],pose['orientation']
  x[0]= position.x
  x[1]= position.y
  x[2]= position.z
  x[3]= orientation.x
  x[4]= orientation.y
  x[5]= orientation.z
  x[6]= orientation.w
  return x

if __name__=='__main__':
  rospy.init_node('baxter_test')
  arm= RIGHT

  rs= baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
  init_state= rs.state().enabled
  def clean_shutdown():
    if not init_state:
      print 'Disabling robot...'
      rs.disable()
  rospy.on_shutdown(clean_shutdown)
  rs.enable()

  limbs= [None,None]
  limbs[RIGHT]= baxter_interface.Limb(LRTostr(RIGHT))
  limbs[LEFT]=  baxter_interface.Limb(LRTostr(LEFT))
  kin= [None,None]
  kin[RIGHT]= baxter_kinematics(LRTostr(RIGHT))
  kin[LEFT]=  baxter_kinematics(LRTostr(LEFT))

  joint_names= [[],[]]
  #joint_names[RIGHT]= ['right_'+joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
  #joint_names[LEFT]=  ['left_' +joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
  joint_names[RIGHT]= limbs[RIGHT].joint_names()
  joint_names[LEFT]=  limbs[LEFT].joint_names()

  viz= TSimpleVisualizer(rospy.Duration())
  viz.viz_frame= 'base'
  def viz_x(x, col):
    viz.AddMarker(x, scale=[0.05,0.05,0.008], alpha=0.8, rgb=viz.ICol(col))
    viz.AddArrow(x, scale=[0.05,0.002,0.002], alpha=0.8, rgb=viz.ICol(col))

  # Set joint position speed ratio for execution.
  # joint position motion speed ratio [0.0-1.0].
  limbs[RIGHT].set_joint_position_speed(0.3)
  limbs[LEFT].set_joint_position_speed(0.3)

  q0=[ 0.70, 0.02,  0.05, 1.51,  1.05, 0.18, -0.41]
  q1=[-0.70, 0.02, -0.05, 1.51, -1.05, 0.18,  0.41]
  angles= {joint:q0[j] for j,joint in enumerate(joint_names[RIGHT])}  #Deserialize
  limbs[RIGHT].move_to_joint_positions(angles, timeout=20.0)
  angles= {joint:q1[j] for j,joint in enumerate(joint_names[LEFT])}  #Deserialize
  limbs[LEFT].move_to_joint_positions(angles, timeout=20.0)
  time.sleep(1.0)

  print 'End Cartesian poses at goal joint positions.'
  angles= {joint:q0[j] for j,joint in enumerate(joint_names[RIGHT])}  #Deserialize
  x= kin[RIGHT].forward_position_kinematics(joint_values=angles)
  viz_x(x, 0)
  print '--RIGHT:',x
  angles= {joint:q1[j] for j,joint in enumerate(joint_names[LEFT])}  #Deserialize
  x= kin[LEFT].forward_position_kinematics(joint_values=angles)
  viz_x(x, 0)
  print '--LEFT:',x

  print 'End Cartesian poses at actual joint positions.'
  x= kin[RIGHT].forward_position_kinematics(joint_values=limbs[RIGHT].joint_angles())
  viz_x(x, 1)
  print '--RIGHT:',x
  x= kin[LEFT].forward_position_kinematics(joint_values=limbs[LEFT].joint_angles())
  viz_x(x, 1)
  print '--LEFT:',x

  print 'End Cartesian poses at actual joint positions obtained through baxter_interface.'
  x= BPoseToX(limbs[RIGHT].endpoint_pose())
  viz_x(x, 2)
  print '--RIGHT:',x
  x= BPoseToX(limbs[LEFT].endpoint_pose())
  viz_x(x, 2)
  print '--LEFT:',x

  #print limbs[RIGHT].

  rospy.signal_shutdown('Done.')
