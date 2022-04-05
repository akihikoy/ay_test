#!/usr/bin/python
#Following a trajectory.

from dxl_fd2f4dof import *
import numpy as np

#Setup the device
gripper= TFD2F4DoF()
gripper.Setup()
gripper.EnableTorque()

#More effort to follow trajectory.
#gripper.SetPWM({jname:e for e,jname in zip([100]*4.,gripper.JointNames())})

q_traj= [
  [ 0.2194, 0.2884, 0.1043, 0.7532 ],
  [ -0.1258, 0.2869, 0.6136, 0.7517 ],
  [ -0.3774, 0.2869, 0.7041, 0.7532 ],
  [ 0.0215, 0.2884, 0.7409, 0.7532 ],
  [ -0.4786, 0.2884, 0.7424, 0.7517 ],
  [ -0.5093, 0.2730, 0.4387, 0.7517 ],
  [ -0.3543, 0.2838, 0.4495, 0.7532 ],
  [ -0.1626, 0.2838, 0.6305, 0.7517 ],
  [ 0.1856, 0.2838, 0.5890, 0.7532 ],
  ]
t_traj= np.array(range(len(q_traj)))*1.0

print q_traj
print t_traj

gripper.FollowTrajectory(gripper.JointNames(), q_traj, t_traj, blocking=True)

#gripper.DisableTorque()
gripper.Quit()
