#!/usr/bin/python3
#\file    joint_spring.py
#\brief   Joint spring controller test
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.29, 2015
'''
NOTE: run beforehand:
  $ rosrun baxter_interface joint_trajectory_action_server.py
'''

from bxtr import *

if __name__=='__main__':
  rospy.init_node('baxter_test')

  EnableBaxter()
  robot= TRobotBaxter()
  robot.Init()

  #Joint springs mode
  robot.ActivateJointSprings(arms=(RIGHT,LEFT), stop_err=0.2, stop_dt=None)

  rospy.signal_shutdown('Done.')

