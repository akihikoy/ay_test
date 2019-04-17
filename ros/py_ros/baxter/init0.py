#!/usr/bin/python
#\file    baynat.py
#\brief   Go to an initial pose.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.07, 2015
'''
NOTE: run beforehand:
  $ rosrun baxter_interface joint_trajectory_action_server.py
'''

from bxtr import *

def GoNatural(robot):
  q0= [robot.Q(RIGHT),robot.Q(LEFT)]
  qnat= [[0.6772525170776368, -0.8617137066101075, -0.1092961310119629, 2.4812139215698243, -0.7577865083496095, -1.4657186411499024, -0.12732040524902344], [-0.5127330777648926, -0.7654564122802735, -0.13767477555541993, 2.5398886866394044, 0.4371845240478516, -1.3917040682189943, 0.3405437345214844]]
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

