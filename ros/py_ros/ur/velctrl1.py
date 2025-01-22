#!/usr/bin/python3
#\file    velctrl1.py
#\brief   Velocity control ver.1 (waiving).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.27, 2018
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
import math,time

if __name__=='__main__':
  rospy.init_node('ur_test')

  pub_vel= rospy.Publisher('/ur_driver/joint_speed', trajectory_msgs.msg.JointTrajectory, queue_size=1)
  joint_names= ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
  traj= trajectory_msgs.msg.JointTrajectory()
  traj.joint_names= joint_names
  traj.points= [trajectory_msgs.msg.JointTrajectoryPoint()]

  t0= time.time()
  rate= rospy.Rate(125)  #UR receives velocities at 125 Hz.

  try:
    while not rospy.is_shutdown():
      t= time.time()-t0
      traj.points[0].velocities= [0.08*math.sin(math.pi*t)]*6
      traj.points[0].accelerations= [10.0]*6
      pub_vel.publish(traj)
      #print traj.points[0].velocities
      rate.sleep()

  except KeyboardInterrupt:
    print('Interrupted')

  finally:
    #To make sure the robot stops:
    traj.points[0].velocities= [0.0]*6
    pub_vel.publish(traj)
