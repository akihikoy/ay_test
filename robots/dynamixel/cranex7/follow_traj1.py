#!/usr/bin/python
#Following a trajectory.

from dxl_cranex7 import *

#Setup the device
crane= TCraneX7()
crane.Setup()
crane.EnableTorque()

#More effort to follow trajectory.
crane.SetPWM({jname:e for e,jname in zip([80]*8,crane.JointNames())})

q_traj= [[0.0]*8,
  [0.0, -0.65, 0.0, -1.92, 0.0, 1.0, 0.0, 0.0],
  [0.0, -0.52, 0.0, -2.38, 0.0, 1.22, 0.0, 0.0]]
t_traj= [8.0,12.0,14.0]

print q_traj
print t_traj

crane.FollowTrajectory(crane.JointNames(), q_traj, t_traj, wait=True)

#crane.DisableTorque()
crane.Quit()
