#!/usr/bin/python3
#Following sine curve with trajectory control.

from dxl_fd2f4dof import *
import time
import math
import numpy as np

#Setup the device
gripper= TFD2F4DoF()
gripper.Setup()
gripper.EnableTorque()

#Move to initial pose
p_start= [0.28, 0.28, 0.72, 0.72]
gripper.MoveTo({jname:p for jname,p in zip(gripper.JointNames(),p_start)})
time.sleep(0.5)
print('Current position=',gripper.Position())

#Generate a trajectory
amp= [-0.1, -0.1, -0.6, -0.6]
angvel= [3, 3, 3, 3]
q_traj= [p_start]
t_traj= [0.0]
for t in np.mgrid[0:2*math.pi:0.05]:
  p_trg= [p0 + a*0.5*(1.0-math.cos(w*t)) for p0,a,w in zip(p_start,amp,angvel)]
  q_traj.append(p_trg)
  t_traj.append(t)

#print q_traj
#print t_traj

gripper.FollowTrajectory(gripper.JointNames(), q_traj, t_traj, blocking=True)

#gripper.DisableTorque()
gripper.Quit()

