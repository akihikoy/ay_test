#!/usr/bin/python
#Move to two target poses.

from dxl_fd2f4dof import *
import numpy as np

#Setup the device
gripper= TFD2F4DoF()
gripper.Setup()
gripper.EnableTorque()

pose= [0.28, 0.28, 0.72, 0.72]
#pose= [0]*5
gripper.MoveTo({jname:p for jname,p in zip(gripper.JointNames(),pose)})

pose= np.array(pose)+[0,0,-0.6,-0.6]
gripper.MoveTo({jname:p for jname,p in zip(gripper.JointNames(),pose)})

#gripper.DisableTorque()
gripper.Quit()
