#!/usr/bin/python
#Move to initial target position with trajectory control.

from dxl_fd2f4dof import *

#Setup the device
gripper= TFD2F4DoF()
gripper.Setup()
gripper.EnableTorque()

pose= [0.28, 0.28, 0.72, 0.72]
gripper.FollowTrajectory(gripper.JointNames(),[pose],[3.0],blocking=True)

#gripper.DisableTorque()
gripper.Quit()
