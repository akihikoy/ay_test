#!/usr/bin/python
#\file    hug2.py
#\brief   Hug motion
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.29, 2015
'''
NOTE: run beforehand:
  $ rosrun baxter_interface joint_trajectory_action_server.py
'''

from bxtr import *

def DoHug(robot):
  q0= [robot.Q(RIGHT),robot.Q(LEFT)]
  qnat= [[-0.8271991389221192, 1.0469418865356446, 1.2060923930969238, 1.057296256842041, -2.7860926028137207, -0.22396119477539064, -1.555456517138672], [0.0038349519653320314, 1.05116033369751, -2.2054808752624515, -0.013038836682128907, -2.7791896892761234, -1.259014730218506, 3.0495538028320315]]
  poses= [
      [0.0, q0[RIGHT], q0[LEFT]],
      [2.0, qnat[RIGHT], qnat[LEFT]],
      [4.0, [-0.7102331039794922, 0.872068076916504, 1.2923788123168947, 1.385568145074463, -2.7385391984436036, -0.49394181313476565, -1.377898241143799], [0.218975757220459, 1.049626352911377, -2.274893505834961, 0.5967185258056641, -2.486582854321289, -1.242140941571045, 3.0495538028320315]],
      [6.0, [-0.513500068157959, 0.5261554096435547, 1.651713811468506, 1.5240099110229492, -2.835563483166504, -0.6749515458984375, -1.3702283372131348], [0.10354370306396485, 0.6235631895629883, -2.2499663180603027, 1.4323545590515139, -2.1322332927246097, -1.0803059686340333, 3.0495538028320315]],
      [8.0, [-0.31906800351562503, 0.5111990969787598, 1.5228594254333496, 1.7042526533935547, -3.039199432525635, -0.757019517956543, -1.4220001887451172], [-0.08858739039916992, 0.8275826341186524, -2.133000283117676, 1.706553624572754, -2.093883773071289, -0.8387039948181153, 3.049937298028565]],
    ]

  for i in range(1,len(poses)):
    robot.ActivateJointSprings(arms=(RIGHT,LEFT), target_angles=poses[i][1:], stop_dt=poses[i][0]-[i-1][0])

  #Joint springs mode
  robot.ActivateJointSprings(arms=(RIGHT,LEFT), stop_err=0.4)

  q0= [robot.Q(RIGHT),robot.Q(LEFT)]
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

  DoHug(robot)

  rospy.signal_shutdown('Done.')
