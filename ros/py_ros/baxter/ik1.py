#!/usr/bin/python3
#\file    ik1.py
#\brief   Baxter: inverse kinematics
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
import time, math, copy

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
  arm= RIGHT
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

  print('\n*** Baxter Description ***\n')
  kin[arm].print_robot_description()
  print('\n*** Baxter KDL Chain - RIGHT ***\n')
  kin[RIGHT].print_kdl_chain()
  print('\n*** Baxter KDL Chain - LEFT ***\n')
  kin[LEFT].print_kdl_chain()
  print('\n')

  #Test 1: IK around fixed angles
  q0=[ 0.40, 0.02,  0.05, 1.51,  1.05, 0.18, -0.41]
  #q0= [0.0]*7
  def FK_IK(q,arm):
    angles= {joint:q[j] for j,joint in enumerate(joint_names[arm])}  #Deserialize
    x= kin[arm].forward_position_kinematics(joint_values=angles)
    #x= x.tolist()
    #x= [0.582583, -0.180819, 0.216003] + [0.03085, 0.9945, 0.0561, 0.0829]
    print('Query q:',q)
    print('x=FK(q):',x)
    Md= lambda q:list(map(AngleMod1,q)) if q is not None else None
    print('IK(x[:3]):',Md(kin[arm].inverse_kinematics(x[:3])))  # position, don't care orientation
    print('IK(x):',Md(kin[arm].inverse_kinematics(x[:3], x[3:])))  # position & orientation
    #FK= lambda q: kin[arm].forward_position_kinematics(joint_values=q.tolist()) if q!=None else None
    def FK(q,arm):
      if q is None:  return None
      angles= {joint:q[j] for j,joint in enumerate(joint_names[arm])}  #Deserialize
      return kin[arm].forward_position_kinematics(joint_values=angles)
    print('FK(IK(x)):',FK(Md(kin[arm].inverse_kinematics(x[:3], x[3:])),arm))
  FK_IK([0.0]*7,arm)
  print('\n')
  FK_IK(q0,arm)


  #Test 1.5: Unsolvable IK
  print('-----------')
  x= [0.5, -0.35, -0.3, -0.48630194, -0.86737689, -0.09892459, 0.03717102]
  print('(solvable) x=',x)
  print('IK(x[:3]):',kin[arm].inverse_kinematics(x[:3]))  # position, don't care orientation
  print('IK(x):',kin[arm].inverse_kinematics(x[:3], x[3:]))  # position & orientation
  x[0]= 1.5
  print('(unsolvable) x=',x)
  print('IK(x[:3]):',kin[arm].inverse_kinematics(x[:3]))  # position, don't care orientation
  print('IK(x):',kin[arm].inverse_kinematics(x[:3], x[3:]))  # position & orientation


  #Test 2: follow a circular trajectory (robot moves!)
  print('-----------')
  rs= baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
  init_state= rs.state().enabled
  def clean_shutdown():
    if not init_state:
      print('Disabling robot...')
      rs.disable()
  rospy.on_shutdown(clean_shutdown)
  rs.enable()

  #q0=[0.65, -0.14, 0.05, 1.60, 3.07, 1.42, 3.06]
  q0=[0.60, -0.17, 0.02, 1.63, -0.20, -1.50, 0.02]
  angles= {joint:q0[j] for j,joint in enumerate(joint_names[arm])}  #Deserialize
  print('Moving to q0=',q0)
  limbs[arm].move_to_joint_positions(angles, timeout=10.0, threshold=0.01)
  print('Done')
  x0= kin[arm].forward_position_kinematics(joint_values=angles)
  print('Initial x0=',x0)
  for i in range(100000):
    if rospy.is_shutdown():  break
    t= i/10.0
    x= copy.deepcopy(x0)
    x[1]= x0[1]+0.1*(math.cos(t)-1.0)
    x[2]= x0[2]+0.1*math.sin(t)
    print(t, x)
    angles= limbs[arm].joint_angles()
    q_seed= [angles[joint] for joint in joint_names[arm]]  #Serialize
    Md= lambda q:list(map(AngleMod1,q)) if q is not None else None
    q= kin[arm].inverse_kinematics(x[:3], x[3:], seed=q_seed)
    #if i%4==0:  print t, '[%s]'%','.join(map(lambda f:'%.3f'%f, list(q) ))
    angles= {joint:q[j] for j,joint in enumerate(joint_names[arm])}  #Deserialize
    limbs[arm].move_to_joint_positions(angles, timeout=0.2, threshold=0.05)

  rospy.signal_shutdown('Done.')
