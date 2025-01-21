#!/usr/bin/python3
#Following sine curve with trajectory control.

from dxl_mikata import *
import time
import math
import numpy as np

#Setup the device
mikata= TMikata()
mikata.Setup()
mikata.EnableTorque()

#Move to initial pose
p_start= [0, 0, 1, -1.3, 0]
mikata.MoveTo({jname:p for jname,p in zip(mikata.JointNames(),p_start)})
time.sleep(0.5)
print('Current position=',mikata.Position())

#Generate a trajectory
gain= [0.45, 0.15, 0.15, 0.7, 0.7]
angvel= [1, 2, 1, 3, 2]
q_traj= [p_start]
t_traj= [0.0]
for t in np.mgrid[0:2*math.pi:0.05]:
  p_trg= [p0 + g*math.sin(w*t) for p0,g,w in zip(p_start,gain,angvel)]
  q_traj.append(p_trg)
  t_traj.append(t)

print(q_traj)
print(t_traj)

mikata.FollowTrajectory(mikata.JointNames(), q_traj, t_traj, blocking=True)

#mikata.DisableTorque()
mikata.Quit()

