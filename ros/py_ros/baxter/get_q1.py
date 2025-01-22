#!/usr/bin/python3
#\file    get_q1.py
#\brief   Baxter: get current joint angles
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
  limbs= [None,None]
  limbs[RIGHT]= baxter_interface.Limb(LRTostr(RIGHT))
  limbs[LEFT]=  baxter_interface.Limb(LRTostr(LEFT))

  joint_names= [[],[]]
  #joint_names[RIGHT]= ['right_'+joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
  #joint_names[LEFT]=  ['left_' +joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
  joint_names[RIGHT]= limbs[RIGHT].joint_names()
  joint_names[LEFT]=  limbs[LEFT].joint_names()

  for i in range(100000):
    if rospy.is_shutdown():  break

    angles= limbs[arm].joint_angles()
    q= [angles[joint] for joint in joint_names[arm]]  #Serialize
    print('@%d, angles=%r'%(i,angles))
    print('@%d, q=%r'%(i,q))
    time.sleep(2.0e-3)


