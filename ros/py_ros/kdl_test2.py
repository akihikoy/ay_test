#!/usr/bin/python
#\file    kdl_test2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1

import numpy as np
from kdl_kin import TKinematics

if __name__=='__main__':
  np.set_printoptions(precision=3)

  print 'Testing TKinematics.'
  print 'Before executing this script, a robot model should be uploaded to robot_description:'
  print '  rosparam load xxxx.urdf robot_description'
  #kin= TKinematics(end_link='link_t')
  kin= TKinematics(end_link='tool0')
  kin.print_robot_description()

  DoF= len(kin.joint_names)
  q0= [0.0]*DoF
  angles= {joint:q0[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
  x0= kin.forward_position_kinematics(angles)
  print 'q0=',q0
  print 'x0= FK(q0)=',x0

  import random
  q1= [3.0*(random.random()-0.5) for j in range(DoF)]
  angles= {joint:q1[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
  x1= kin.forward_position_kinematics(angles)
  print 'q1=',q1
  print 'x1= FK(q1)=',x1

  seed= [0.0]*DoF
  #seed= [3.0*(random.random()-0.5) for j in range(DoF)]
  q2= kin.inverse_kinematics(x1[:3], x1[3:], seed=seed, maxiter=2000, eps=1.0e-4)  #, maxiter=500, eps=1.0e-6
  print 'q2= IK(x1)=',q2
  if q2 is not None:
    angles= {joint:q2[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
    x2= kin.forward_position_kinematics(angles)
    print 'x2= FK(q2)=',x2
    print 'x2==x1?', np.allclose(x2,x1)
    print '|x2-x1|=',np.linalg.norm(x2-x1)
  else:
    print 'Failed to solve IK.'
