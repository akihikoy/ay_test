#!/usr/bin/python3
#Following a trajectory.

from dxl_mikata6 import *

#Setup the device
mikata= TMikata6()
mikata.Setup()
mikata.EnableTorque()

#More effort to follow trajectory.
mikata.SetPWM({jname:e for e,jname in zip([80]*8,mikata.JointNames())})

q_traj= [[0.0]*7,
  [0.0, -1.0, 0.78, 0.0, 0.35, 0.0, 0.2],
  [0.2, -1.0, 0.78, 0.0, 0.86, 0.5, -0.2]]
t_traj= [8.0,12.0,14.0]

print(q_traj)
print(t_traj)

mikata.FollowTrajectory(mikata.JointNames(), q_traj, t_traj, blocking=True)

#mikata.DisableTorque()
mikata.Quit()
