#!/usr/bin/python3
#Following a trajectory.

from dxl_xw540_xw430 import *
import numpy as np

#Setup the device
device= TXW540XW430()
device.Setup()
device.EnableTorque()

#More effort to follow trajectory.
#device.SetPWM({jname:e for e,jname in zip([100]*4.,device.JointNames())})

q_traj= [
  [ -0.7578, -1.3744 ],
  [ -0.7578, -1.3744 ],
  [ -0.6550, -1.3744 ],
  [ -0.6550, 1.9190 ],
  [ -0.6550, 1.9190 ],
  [ -0.6550, -1.3744 ],
  [ -0.3850, -1.3744 ],
  [ -0.7578, -1.3744 ],
  ]
t_traj= np.array(list(range(len(q_traj))))*1.0

print(q_traj)
print(t_traj)

device.FollowTrajectory(device.JointNames(), q_traj, t_traj, blocking=True)

#device.DisableTorque()
device.Quit()
