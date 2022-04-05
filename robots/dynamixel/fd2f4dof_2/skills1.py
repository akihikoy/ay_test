#!/usr/bin/python
#Running predefined skills by pressing a key.
from dxl_fd2f4dof import *
import time
from kbhit2 import TKBHit

#Setup the device
gripper= TFD2F4DoF()
gripper.Setup()
gripper.EnableTorque()

skills= {
  'a': dict(
    #q_traj= [
      #[0.27765052260730105, 0.2791845033951867, -0.31600004230444206, 0.7194369895183657],
      #[0.1595340019401067, 0.2807184841830723, 0.4924078329112908, 0.7209709703062513],
      #[0.12578642460662257, 0.2807184841830723, 0.7286408742456796, 0.7209709703062513],
      #[0.27765052260730105, 0.2791845033951867, 0.7286408742456796, 0.7209709703062513],
      ##[0.27765052260730105, 0.2791845033951867, -0.31600004230444206, 0.7194369895183657],
      #],
    #t_traj= [1.0,2.0,3.0,4.0] ),
    q_traj= [
      [0.2807184841830723, 0.2899223689103862, 0.7102331047910518, 0.6964272777000811],
      [0.2807184841830723, -0.1641359443037636, 0.7086991240031663, 0.679553489033339],
      [0.2807184841830723, 0.29605829206192874, 0.7332428166093364, 0.7102331047910518],
      ],
    t_traj= [1.0,2.0,3.0] ),
  }

print skills

try:
  kbhit= TKBHit()
  while True:
    c= kbhit.KBHit()
    if c=='q':  break
    elif c is not None:
      gripper.FollowTrajectory(gripper.JointNames(), skills['a']['q_traj'], skills['a']['t_traj'], blocking=True)
    time.sleep(0.0025)
except KeyboardInterrupt:
  pass

#gripper.DisableTorque()
gripper.Quit()

