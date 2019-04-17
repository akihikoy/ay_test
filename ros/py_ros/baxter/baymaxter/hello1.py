#!/usr/bin/python
#\file    hello1.py
#\brief   Baymaxter hello motion
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.29, 2015
'''
NOTE: run beforehand:
  $ rosrun baxter_interface joint_trajectory_action_server.py
'''

from bxtr import *

def DoHello(robot, sw=False):
  q0= [robot.Q(RIGHT),robot.Q(LEFT)]
  qnat= [[-0.8271991389221192, 1.0469418865356446, 1.2060923930969238, 1.057296256842041, -2.7860926028137207, -0.22396119477539064, -1.555456517138672], [0.0038349519653320314, 1.05116033369751, -2.2054808752624515, -0.013038836682128907, -2.7791896892761234, -1.259014730218506, 3.0495538028320315]]
  if sw==False:
    qvia1= [[-0.8241311773498535, 1.047708876928711, 1.2053254027038576, 1.0599807232177736, -2.784942117224121, -0.22549517556152346, -1.5546895267456056], [0.002684466375732422, 1.0469418865356446, -2.205097380065918, -0.013038836682128907, -2.7795731844726563, -1.2601652158081056, 3.0495538028320315]]
    qhello= [[-0.7263399022338868, 1.0097428524719239, 2.0831459075683596, 2.068573090100098, -2.655704235992432, -0.947616630633545, -1.9976264787414553], [0.0034514567687988283, 1.0473253817321777, -2.205097380065918, -0.013038836682128907, -2.77880619407959, -1.259398225415039, 3.049937298028565]]
  else:
    qvia1= [[-0.8264321485290528, 1.0480923721252442, 1.2060923930969238, 1.057296256842041, -2.785709107617188, -0.22587867075805665, -1.5558400123352052], [-0.2304806131164551, 1.05116033369751, -2.213917769586182, 0.004218447161865235, -2.7435246359985355, -1.321524447253418, 3.049170307635498]]
    qhello= [[-0.828733119708252, 1.047708876928711, 1.2057088979003907, 1.057296256842041, -2.786476098010254, -0.22319420438232423, -1.5550730219421387], [-0.2419854690124512, 1.0105098428649903, -2.699039193200684, 1.8097138324401856, -2.856272223779297, -1.0538448000732423, 3.0453353556701663]]
  poses1= [
      [0.0, q0[RIGHT], q0[LEFT]],
      [1.0, qnat[RIGHT], qnat[LEFT]],
      [3.0, qvia1[RIGHT], qvia1[LEFT]],
      [6.0, qhello[RIGHT], qhello[LEFT]],
    ]
  poses2= [
      [1.0, qhello[RIGHT], qhello[LEFT]],
      #[5.0, [-0.8417719563903809, 1.0473253817321777, 1.5151895215026856, 1.040038972998047, -3.038815937329102, -0.9292088611999513, -1.9427866656372073], [0.0034514567687988283, 1.0469418865356446, -2.2058643704589844, -0.012655341485595705, -2.7791896892761234, -1.2567137590393067, 3.049937298028565]],
      [6.0, qnat[RIGHT], qnat[LEFT]],
    ]

  t_traj,qr_traj,ql_traj= DecomposePoseTraj(poses1)
  client_r= robot.FollowQTraj(qr_traj,t_traj,arm=RIGHT,blocking=False)
  client_l= robot.FollowQTraj(ql_traj,t_traj,arm=LEFT,blocking=False)
  client_r.wait_for_result(timeout=rospy.Duration(t_traj[-1]+5.0))
  client_l.wait_for_result(timeout=rospy.Duration(t_traj[-1]+5.0))
  print client_r.get_result(), client_l.get_result()

  time.sleep(1.0)
  robot.NodHead()
  robot.NodHead()
  time.sleep(2.0)

  t_traj,qr_traj,ql_traj= DecomposePoseTraj(poses2)
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

  DoHello(robot)
  DoHello(robot,True)

  rospy.signal_shutdown('Done.')

