#!/usr/bin/python3
#Move to two target poses.

from dxl_cranex7 import *

#Setup the device
crane= TCraneX7()
crane.Setup()
crane.EnableTorque()

pose= [0.0]*8
crane.FollowTrajectory(crane.JointNames(),[pose],[8.0],blocking=True)

pose= [0.0, -0.65, 0.0, -1.92, 0.0, 1.0, 0.0, 0.0]
crane.FollowTrajectory(crane.JointNames(),[pose],[8.0],blocking=True)

pose= [0.0, -0.52, 0.0, -2.38, 0.0, 1.22, 0.0, 0.0]
crane.MoveTo({jname:p for jname,p in zip(crane.JointNames(),pose)})

#crane.DisableTorque()
crane.Quit()
