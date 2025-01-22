#!/usr/bin/python3
#\file    key_velctrl1.py
#\brief   Velocity control with keyboard.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.12, 2018
'''
We control UR by joint velocities.
Using /ur_driver/joint_speed topic
whose type is trajectory_msgs/JointTrajectory
NOTE that only msg.points[0].velocities and msg.points[0].accelerations are used.
NOTE that only max(100,max(msg.points[0].accelerations)) is used.
'''

import roslib
import rospy
import trajectory_msgs.msg

from kbhit2 import TKBHit
import threading
import sys,time

def ReadKeyboard(is_running, key_cmd, key_locker):
  kbhit= TKBHit()
  dt_hold= 0.1
  t_prev= 0
  while is_running[0]:
    c= kbhit.KBHit()
    if c is not None or time.time()-t_prev>dt_hold:
      with key_locker:
        key_cmd[0]= c
      t_prev= time.time()
    time.sleep(0.0025)

if __name__=='__main__':
  rospy.init_node('ur_test')

  pub_vel= rospy.Publisher('/ur_driver/joint_speed', trajectory_msgs.msg.JointTrajectory, queue_size=1)
  joint_names= ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
  traj= trajectory_msgs.msg.JointTrajectory()
  traj.joint_names= joint_names
  traj.points= [trajectory_msgs.msg.JointTrajectoryPoint()]

  #Start a thread for reading keyboard
  key_cmd= [None]
  key_locker= threading.RLock()
  is_running= [True]
  t1= threading.Thread(name='t1', target=lambda a1=is_running,a2=key_cmd,a3=key_locker: ReadKeyboard(a1,a2,a3))
  t1.start()

  rate= rospy.Rate(125)  #UR receives velocities at 125 Hz.

  while not rospy.is_shutdown():
    with key_locker:
      c= key_cmd[0]
      #key_cmd[0]= None

    traj.points[0].velocities= [0.0]*6

    if c is not None:
      if c=='q':  break
      if c in ('z','Z','x','X'):
        if   c=='z':  traj.points[0].velocities= [0.05]*6
        elif c=='Z':  traj.points[0].velocities= [0.20]*6
        elif c=='x':  traj.points[0].velocities= [-0.05]*6
        elif c=='X':  traj.points[0].velocities= [-0.20]*6
        traj.points[0].accelerations= [10.0]*6

    pub_vel.publish(traj)
    print(traj.points[0].velocities)
    rate.sleep()

  #To make sure the robot stops:
  traj.points[0].velocities= [0.0]*6
  pub_vel.publish(traj)

  #Stop the reading keyboard thread
  is_running[0]= False
  t1.join()
