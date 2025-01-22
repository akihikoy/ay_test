#!/usr/bin/python3
#\file    ik1.py
#\brief   IK for Mikata arm.
#         As Mikata arm has 4 joints, we use a weighted pinv of Jacobian.
#         Comparing my IK and KDL::IK.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.05, 2018

import numpy as np
import numpy.linalg as la
from kdl_kin2 import TKinematics

import roslib; roslib.load_manifest('ay_py')
import rospy
from ay_py.core import *

'''
kin: TKinematics object.
w_x: Weights on error of x. e.g. [1.0,1.0,1.0, 0.01,0.01,0.01]
with_st: With IK status [True/False, q].
'''
def InverseKinematics(kin, position, orientation=None, seed=None, min_joints=None, max_joints=None, maxiter=500, eps=1.0e-6, w_x=None, with_st=False):

  alpha= 0.3
  eps_std= eps*1.0e-2

  q= np.array(seed)
  if orientation is None:
    x_goal= np.array(position)
    pos_ik= True
  else:
    x_goal= np.array(list(position)+list(orientation))
    pos_ik= False

  if w_x is not None:
    if pos_ik:
      Wx= np.diag(w_x[:3])
    else:
      Wx= np.diag(w_x)
  else:
    Wx= None

  solved= False
  err= [100.0,90.0]
  for i in range(maxiter):
    angles= {joint:q[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
    x= kin.forward_position_kinematics(angles)
    J= kin.jacobian(angles)

    if pos_ik:
      x_err= np.array([x_goal[0]-x[0],x_goal[1]-x[1],x_goal[2]-x[2]])
      J= J[:3,:]
    else:
      x_err= np.array(DiffX(x, x_goal))

    #if Wx is not None:
      #err.append(la.norm(np.dot(Wx,x_err)))
    #else:
      #err.append(la.norm(x_err))
    err.append(la.norm(x_err))

    if len(err)>10:  err.pop(0)
    #print i,err[-1],np.std(err)
    if err[-1]<eps:
      solved= True
      break
    if np.std(err)<eps_std:
      break

    if Wx is not None:
      dq= np.dot(la.pinv(Wx*J)*Wx, x_err)
    else:
      dq= np.dot(la.pinv(J), x_err)
    dq= np.asarray(dq).ravel()

    q= q + alpha*dq

  if not with_st:
    if solved:  return q
    else:  return None
  else:
    if solved:  return True,q
    else:  return False,q


if __name__=='__main__':
  np.set_printoptions(precision=3)

  print('Testing TKinematics (robot_description == Mikata Arm is assumed).')
  print('Before executing this script, run:')
  print('  rosparam load `rospack find mikata_arm_description`/description/urdf/mikata_arm_4.urdf robot_description')
  kin= TKinematics(base_link='base_link', end_link='link_5')
  #kin.print_robot_description()

  DoF= 4
  #q0= [0.0]*DoF
  #angles= {joint:q0[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
  #x0= kin.forward_position_kinematics(angles)
  #J0= kin.jacobian(angles)
  #print 'q0=',q0
  #print 'x0= FK(q0)=',x0
  #print 'J0= J(q0)=',J0

  import random
  q1= [3.0*(random.random()-0.5) for j in range(DoF)]
  angles= {joint:q1[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
  x1= kin.forward_position_kinematics(angles)
  #J1= kin.jacobian(angles)
  print('q1=',q1)
  #print 'x1= FK(q1)=',x1
  x1= x1+np.array([0.04,0.04,-0.03, 0.0,0.0,0.0,0.0])
  print('x1=',x1)
  #print 'J1= J(q1)=',J1

  #COMPARING MY IK vs. KDL::IK
  seed= [0.0]*DoF
  w_x= [1.0,1.0,1.0, 0.01,0.01,0.01]
  #seed= [3.0*(random.random()-0.5) for j in range(DoF)]
  t0= time.time()
  res2a,q2a= kin.inverse_kinematics(x1[:3], x1[3:], seed=seed, maxiter=2000, eps=1.0e-4,
                                    w_x=np.diag(w_x).tolist(), with_st=True)
  print('IK(a) took:',time.time()-t0)
  t0= time.time()
  res2b,q2b= InverseKinematics(kin, x1[:3], x1[3:], seed=seed, maxiter=2000, eps=1.0e-4,
                               w_x=w_x,with_st=True)
  print('IK(b) took:',time.time()-t0)
  print('')
  if not res2a:  print('Failed to solve IK(a).')
  print('q2a= IK(x1)=',q2a)
  if q2a is not None:
    angles= {joint:q2a[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
    x2a= kin.forward_position_kinematics(angles)
    print('x2a= FK(q2a)=',x2a)
    print('x2a==x1?', np.allclose(x2a,x1))
    print('x2a-x1=',np.array(DiffX(x1, x2a)))
    print('w_x|x2a-x1|=',np.linalg.norm(np.dot(np.diag(w_x),DiffX(x1, x2a))))
  else:
    print('Failed to solve IK(a).')
  print('')
  if not res2b:  print('Failed to solve IK(b).')
  print('q2b= IK(x1)=',q2b)
  if q2b is not None:
    angles= {joint:q2b[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
    x2b= kin.forward_position_kinematics(angles)
    print('x2b= FK(q2b)=',x2b)
    print('x2b==x1?', np.allclose(x2b,x1))
    print('x2b-x1=',np.array(DiffX(x1, x2b)))
    print('w_x|x2b-x1|=',np.linalg.norm(np.dot(np.diag(w_x),DiffX(x1, x2b))))
  else:
    print('Failed to solve IK(b).')
