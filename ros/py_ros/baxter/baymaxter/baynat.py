#!/usr/bin/python
#\file    baynat.py
#\brief   Baymaxter natural pose
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.29, 2015
'''
NOTE: run beforehand:
  $ rosrun baxter_interface joint_trajectory_action_server.py
'''

from bxtr import *

def GoNatural(robot):
  q0= [robot.Q(RIGHT),robot.Q(LEFT)]
  qnat= [[-0.8271991389221192, 1.0469418865356446, 1.2060923930969238, 1.057296256842041, -2.7860926028137207, -0.22396119477539064, -1.555456517138672], [0.0038349519653320314, 1.05116033369751, -2.2054808752624515, -0.013038836682128907, -2.7791896892761234, -1.259014730218506, 3.0495538028320315]]
  poses= [
      [0.0, q0[RIGHT], q0[LEFT]],
      [5.0, qnat[RIGHT], qnat[LEFT]],
    ]

  t_traj,qr_traj,ql_traj= DecomposePoseTraj(poses)
  client_r= robot.FollowQTraj(qr_traj,t_traj,arm=RIGHT,blocking=False)
  client_l= robot.FollowQTraj(ql_traj,t_traj,arm=LEFT,blocking=False)
  client_r.wait_for_result(timeout=rospy.Duration(t_traj[-1]+5.0))
  client_l.wait_for_result(timeout=rospy.Duration(t_traj[-1]+5.0))
  print client_r.get_result(), client_l.get_result()

if __name__=='__main__':
  rospy.init_node('baxter_test')

  EnableBaxter()
  robot= TRobotBaxter()
  robot.Init()

  GoNatural(robot)

  rospy.signal_shutdown('Done.')

