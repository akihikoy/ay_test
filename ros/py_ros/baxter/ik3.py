#!/usr/bin/python
#\file    ik3.py
#\brief   Baxter: inverse kinematics (we check if the joint limits are considered)
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.05, 2016
'''
  Ref.
  ~/catkin_ws/src/baxter_pykdl/src/baxter_pykdl/baxter_pykdl.py
  ~/catkin_ws/src/baxter_pykdl/scripts/baxter_kinematics.py

  NOTE:
  In the latest commit 8b95af3 of the master branch of baxter_pykdl,
  the joint limits are not considered in IK.
  In the following program, joint angle of 'w1' (6th element) in [-1.0, -0.39, 0.18, 2.3, -0.5, -1.9, 0.1]
  (i.e. -1.9) exceeds its limit, but IK does not take it into account.
  TODO:
  To fix this issue, edit:
    catkin_ws/src/baxter_pykdl/src/baxter_pykdl/baxter_pykdl.py
    replace
      self._ik_p_kdl = PyKDL.ChainIkSolverPos_NR
    by
      PyKDL.ChainIkSolverPos_NR_JL
    and change IK execution codes.
    Joint limits can be obtained from URDF:
      jnt = urdf.joints[jnt_name]
      jnt.safety.lower
      jnt.safety.upper
      jnt.limits.lower
      jnt.limits.upper
    For more details, refer to:
    catkin_ws/src/baxter_pykdl/src/baxter_kdl/kdl_kinematics.py
'''

import roslib
import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics
import time, math, copy, random

RIGHT=0
LEFT=1
def LRTostr(whicharm):
  if whicharm==RIGHT: return 'right'
  if whicharm==LEFT:  return 'left'
  return None

# Matlab-like mod function that returns always positive
def Mod(x, y):
  if y==0:  return x
  return x-y*math.floor(x/y)

#Convert radian to [-pi,pi)
def AngleMod1(q):
  return Mod(q+math.pi,math.pi*2.0)-math.pi

if __name__=='__main__':
  rospy.init_node('baxter_test')
  arm= LEFT
  limbs= [None,None]
  limbs[RIGHT]= baxter_interface.Limb(LRTostr(RIGHT))
  limbs[LEFT]=  baxter_interface.Limb(LRTostr(LEFT))
  kin= [None,None]
  kin[RIGHT]= baxter_kinematics(LRTostr(RIGHT))
  kin[LEFT]=  baxter_kinematics(LRTostr(LEFT))  #tip_link=_gripper(default),_wrist,_hand

  joint_names= [[],[]]
  #joint_names[RIGHT]= ['right_'+joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
  #joint_names[LEFT]=  ['left_' +joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
  joint_names[RIGHT]= limbs[RIGHT].joint_names()
  joint_names[LEFT]=  limbs[LEFT].joint_names()

  print '\n*** Baxter Description ***\n'
  kin[arm].print_robot_description()
  print '\n*** Baxter KDL Chain - RIGHT ***\n'
  kin[RIGHT].print_kdl_chain()
  print '\n*** Baxter KDL Chain - LEFT ***\n'
  kin[LEFT].print_kdl_chain()
  print '\n'

  #Test 1: IK around fixed angles
  q0= [-1.0, -0.39, 0.18, 2.3, -0.5, -1.9, 0.1]
  def FK_IK(q,arm):
    angles= {joint:q[j] for j,joint in enumerate(joint_names[arm])}  #Deserialize
    x= kin[arm].forward_position_kinematics(joint_values=angles)
    #x= x.tolist()
    #x= [0.582583, -0.180819, 0.216003] + [0.03085, 0.9945, 0.0561, 0.0829]
    print 'Query q:',q
    print 'x=FK(q):',x.tolist()
    q_seed= [angle + (random.random()-0.5) for angle in q]  #Serialize
    Md= lambda q:map(AngleMod1,q) if q is not None else None
    print 'IK(x):',Md(kin[arm].inverse_kinematics(x[:3], x[3:], seed=q_seed))  # position & orientation
    #FK= lambda q: kin[arm].forward_position_kinematics(joint_values=q.tolist()) if q!=None else None
    def FK(q,arm):
      if q is None:  return None
      angles= {joint:q[j] for j,joint in enumerate(joint_names[arm])}  #Deserialize
      return kin[arm].forward_position_kinematics(joint_values=angles)
    print 'FK(IK(x)):',FK(Md(kin[arm].inverse_kinematics(x[:3], x[3:])),arm).tolist()
  FK_IK(q0,arm)

  rospy.signal_shutdown('Done.')
