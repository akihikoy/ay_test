#!/usr/bin/python
#\file    kdl_test2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.22, 2018

import numpy as np
from kdl_kin2 import TKinematics

if __name__=='__main__':
  np.set_printoptions(precision=3)

  print 'Testing TKinematics (robot_description == Mikata Arm is assumed).'
  print 'Before executing this script, run:'
  print '  rosparam load `rospack find mikata_arm_description`/description/urdf/mikata_arm_4.urdf robot_description'
  kin= TKinematics(base_link='base_link', end_link='link_5')
  kin.print_robot_description()

  DoF= len(kin.joint_names)
  q0= [0.0]*DoF
  angles= {joint:q0[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
  x0= kin.forward_position_kinematics(angles)
  J0= kin.jacobian(angles)
  print 'q0=',q0
  print 'x0= FK(q0)=',x0
  print 'J0= J(q0)=',J0

  import random
  q1= [3.0*(random.random()-0.5) for j in range(DoF)]
  angles= {joint:q1[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
  x1= kin.forward_position_kinematics(angles)
  J1= kin.jacobian(angles)
  print 'q1=',np.array(q1)
  print 'x1= FK(q1)=',x1
  print 'J1= J(q1)=',J1

  seed= [0.0]*DoF
  #seed= [3.0*(random.random()-0.5) for j in range(DoF)]
  w_x= [1.0,1.0,1.0, 0.1,0.1,0.1]
  q2= kin.inverse_kinematics(x1[:3], x1[3:], seed=seed, w_x=np.diag(w_x).tolist(), maxiter=2000, eps=1.0e-4)  #, maxiter=500, eps=1.0e-6
  print 'q2= IK(x1)=',q2
  if q2 is not None:
    angles= {joint:q2[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
    x2= kin.forward_position_kinematics(angles)
    print 'x2= FK(q2)=',x2
    print 'x2==x1?', np.allclose(x2,x1)
    print '|x2-x1|=',np.linalg.norm(x2-x1)
  else:
    print 'Failed to solve IK.'
