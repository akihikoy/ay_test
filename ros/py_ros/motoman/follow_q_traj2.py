#!/usr/bin/python3
#\file    follow_q_traj2.py
#\brief   Following joint angle trajectory.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.22, 2017

import roslib; roslib.load_manifest('motoman_driver')
import rospy
import sensor_msgs.msg
import trajectory_msgs.msg
import copy


#cf. ay_py.ros.base

'''Support function to generate trajectory_msgs/JointTrajectoryPoint.
    q: Joint positions, t: Time from start, dq: Joint velocities.'''
def ROSGetJTP(q,t,dq=None):
  jp= trajectory_msgs.msg.JointTrajectoryPoint()
  jp.positions= q
  jp.time_from_start= rospy.Duration(t)
  if dq is not None:  jp.velocities= dq
  return jp

'''Get trajectory_msgs/JointTrajectory from a joint angle trajectory.
  joint_names: joint names.
  q_traj: joint angle trajectory [q0,...,qD]*N.
  t_traj: corresponding times in seconds from start [t1,t2,...,tN].
  dq_traj: corresponding velocity trajectory [dq0,...,dqD]*N. '''
def ToROSTrajectory(joint_names, q_traj, t_traj, dq_traj=None):
  assert(len(q_traj)==len(t_traj))
  if dq_traj is not None:  (len(dq_traj)==len(t_traj))
  traj= trajectory_msgs.msg.JointTrajectory()
  traj.joint_names= joint_names
  if dq_traj is not None:
    traj.points= [ROSGetJTP(q,t,dq) for q,t,dq in zip(q_traj, t_traj, dq_traj)]
  else:
    traj.points= [ROSGetJTP(q,t) for q,t in zip(q_traj, t_traj)]
  traj.header.stamp= rospy.Time.now()
  return traj


#Wait for subscribers (src: motoman_driver/move_to_joint.py)
def WaitForSubscribers(pub, timeout, num_subs=1):
  time_end= rospy.Time.now()+rospy.Duration(timeout)
  rate= rospy.Rate(10)
  while all((pub.get_num_connections()<num_subs, rospy.Time.now()<time_end, not rospy.is_shutdown())):
    rate.sleep()
  return (pub.get_num_connections()>=num_subs)


if __name__=='__main__':
  rospy.init_node('motoman_test')

  pub_traj= rospy.Publisher('/joint_path_command', trajectory_msgs.msg.JointTrajectory, queue_size=1)
  if not WaitForSubscribers(pub_traj, 3.0):
    print('WARNING: No subscribers of /joint_path_command')

  joint_names= ['joint_'+jkey for jkey in ('s','l','e','u','r','b','t')]
  joint_names= rospy.get_param('controller_joint_names')

  t_traj= [4.0, 6.0, 8.0, 12.0]
  q_traj= [[0.0]*7,
           [0.1,  -0.3,  0.15, -0.7,  0.1,  -0.3,  0.15],
           [0.21, -0.59, 0.30, -1.46, 0.35, -0.68, 0.31],
           [0.0]*7]
  dq_traj= [[0.0]*7]*4
  traj= ToROSTrajectory(joint_names, q_traj, t_traj, dq_traj)

  pub_traj.publish(traj)

  rospy.signal_shutdown('Done.')
