#!/usr/bin/python
#\file    kdl_test3.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1

import numpy as np
from kdl_kin2 import TKinematics

if __name__=='__main__':
  np.set_printoptions(precision=3)

  print 'Testing TKinematics (robot_description == Yaskawa Motoman is assumed).'
  print 'Before executing this script, run:'
  print '  rosparam load `rospack find motoman_sia10f_support`/urdf/sia10f.urdf robot_description'
  kin= TKinematics(end_link='link_t')
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
  print 'q1=',np.array(q1)
  print 'x1= FK(q1)=',x1

  seed= [0.0]*DoF
  #seed= [3.0*(random.random()-0.5) for j in range(DoF)]
  w_q= [1.0]*7; w_q[3]= 0.0
  res2a,q2a= kin.inverse_kinematics(x1[:3], x1[3:], seed=seed, maxiter=2000, eps=1.0e-4, with_st=True)  #, maxiter=500, eps=1.0e-6
  #seed= [s+0.01*(random.random()-0.5) for s in seed]
  #res2b,q2b= kin.inverse_kinematics(x1[:3], x1[3:], seed=seed, maxiter=2000, eps=1.0e-4, with_st=True)  #, maxiter=500, eps=1.0e-6
  res2b,q2b= kin.inverse_kinematics(x1[:3], x1[3:], seed=seed, w_q=np.diag(w_q).tolist(), maxiter=2000, eps=1.0e-4, with_st=True)  #, maxiter=500, eps=1.0e-6
  print ''
  if not res2a:  print 'Failed to solve IK.'
  print 'q2a= IK(x1)=',q2a
  if q2a is not None:
    angles= {joint:q2a[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
    x2a= kin.forward_position_kinematics(angles)
    print 'x2a= FK(q2a)=',x2a
    print 'x2a==x1?', np.allclose(x2a,x1)
    print '|x2a-x1|=',np.linalg.norm(x2a-x1)
  else:
    print 'Failed to solve IK.'
  print ''
  if not res2b:  print 'Failed to solve IK.'
  print 'q2b= IK(x1)=',q2b
  if q2b is not None:
    angles= {joint:q2b[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
    x2b= kin.forward_position_kinematics(angles)
    print 'x2b= FK(q2b)=',x2b
    print 'x2b==x1?', np.allclose(x2b,x1)
    print '|x2b-x1|=',np.linalg.norm(x2b-x1)
  else:
    print 'Failed to solve IK.'
