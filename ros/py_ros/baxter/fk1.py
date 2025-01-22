#!/usr/bin/python3
#\file    fk1.py
#\brief   Baxter: forwrad kinematics
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.08, 2015
'''
  Ref.
  ~/catkin_ws/src/baxter_pykdl/src/baxter_pykdl/baxter_pykdl.py
  ~/catkin_ws/src/baxter_pykdl/scripts/baxter_kinematics.py
'''

import roslib
import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics
import time, math

RIGHT=0
LEFT=1
def LRTostr(whicharm):
  if whicharm==RIGHT: return 'right'
  if whicharm==LEFT:  return 'left'
  return None

if __name__=='__main__':
  rospy.init_node('baxter_test')
  arm= RIGHT
  limbs= [None,None]
  limbs[RIGHT]= baxter_interface.Limb(LRTostr(RIGHT))
  limbs[LEFT]=  baxter_interface.Limb(LRTostr(LEFT))
  kin= [None,None]
  kin[RIGHT]= baxter_kinematics(LRTostr(RIGHT), tip_link='_gripper')
  kin[LEFT]=  baxter_kinematics(LRTostr(LEFT), tip_link='_gripper')  #tip_link=_gripper(default),_wrist,_hand

  joint_names= [[],[]]
  #joint_names[RIGHT]= ['right_'+joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
  #joint_names[LEFT]=  ['left_' +joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
  joint_names[RIGHT]= limbs[RIGHT].joint_names()
  joint_names[LEFT]=  limbs[LEFT].joint_names()

  print('\n*** Baxter Description ***\n')
  kin[arm].print_robot_description()
  print('\n*** Baxter KDL Chain - RIGHT ***\n')
  kin[RIGHT].print_kdl_chain()
  print('\n*** Baxter KDL Chain - LEFT ***\n')
  kin[LEFT].print_kdl_chain()
  print('\n')

  #Test 1: FK of fixed angles (symmetry)
  q0=[ 0.40, 0.02,  0.05, 1.51,  1.05, 0.18, -0.41]
  q1=[-0.40, 0.02, -0.05, 1.51, -1.05, 0.18,  0.41]
  #TEST:
  #q0= [0.0]*7
  #q1= [0.0]*7
  angles= {joint:q0[j] for j,joint in enumerate(joint_names[RIGHT])}  #Deserialize
  x= kin[RIGHT].forward_position_kinematics(joint_values=angles)
  print(x)
  angles= {joint:q1[j] for j,joint in enumerate(joint_names[LEFT])}  #Deserialize
  x= kin[LEFT].forward_position_kinematics(joint_values=angles)
  print(x)
  print(type(x))
  print(x.shape)

  ##Test 2: FK around fixed angles (right arm)
  #q=[0.40, 0.02, 0.05, 1.51, 1.05, 0.18, -0.41]
  #for i in xrange(100000):
    #if rospy.is_shutdown():  break
    #q[0]= 0.40 + 1.0*math.sin(i/100.0)
    #angles= {joint:q[j] for j,joint in enumerate(joint_names[arm])}  #Deserialize
    ##print i, angles
    #print i, kin[arm].forward_position_kinematics(joint_values=angles)

  #Test 3: FK of current angles (right arm)
  for i in range(100000):
    if rospy.is_shutdown():  break
    angles= limbs[arm].joint_angles()
    print(i, kin[arm].forward_position_kinematics(joint_values=angles))

