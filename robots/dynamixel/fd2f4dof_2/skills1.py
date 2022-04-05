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
      [ 0.0706, 0.2915, 0.3682, 0.7517 ],
      [ -0.1197, 0.2884, 0.5768, 0.7532 ],
      [ -0.2546, 0.2869, 0.4449, 0.7517 ],
      [ 0.0936, 0.2869, 0.7409, 0.7517 ],
      [ 0.0706, 0.2915, 0.3682, 0.7517 ],
      ],
    t_traj= [1.0,2.0,3.0,4.0,5.0] ),
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

