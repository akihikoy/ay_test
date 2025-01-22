#!/usr/bin/python3
#\file    follow_q_traj3.py
#\brief   Following joint angle trajectory
#         where target velocity is automatically decided with spline.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.22, 2017

import roslib; roslib.load_manifest('motoman_driver')
import rospy
import sensor_msgs.msg
import trajectory_msgs.msg
import copy

from cubic_hermite_spline import TCubicHermiteSpline

from follow_q_traj2 import ToROSTrajectory

'''Convert joint angle trajectory to joint velocity trajectory.'''
def QTrajToDQTraj(q_traj, t_traj):
  dof= len(q_traj[0])

  #Modeling the trajectory with spline.
  splines= [TCubicHermiteSpline() for d in range(dof)]
  for d in range(len(splines)):
    data_d= [[t,q[d]] for q,t in zip(q_traj,t_traj)]
    splines[d].Initialize(data_d, tan_method=splines[d].CARDINAL, c=0.0, m=0.0)

  #NOTE: We don't have to make spline models as we just want velocities at key points.
  #  They can be obtained by computing tan_method, which will be more efficient.

  dq_traj= []
  for t in t_traj:
    dq= [splines[d].Evaluate(t,with_tan=True)[1] for d in range(dof)]
    dq_traj.append(dq)
  return dq_traj


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
  dq_traj= QTrajToDQTraj(q_traj, t_traj)
  print(dq_traj)
  traj= ToROSTrajectory(joint_names, q_traj, t_traj, dq_traj)

  pub_traj.publish(traj)

  rospy.signal_shutdown('Done.')
