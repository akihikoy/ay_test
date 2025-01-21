#!/usr/bin/python3
#Move to two target poses.

from dxl_mikata6 import *

#Setup the device
mikata= TMikata6()
mikata.Setup()
mikata.EnableTorque()

pose= [0.0]*7
mikata.FollowTrajectory(mikata.JointNames(),[pose],[8.0],blocking=True)

pose= [0.0, -1.0, 0.78, 0.0, 0.35, 0.0, 0.2]
mikata.FollowTrajectory(mikata.JointNames(),[pose],[8.0],blocking=True)

pose= [0.2, -1.0, 0.78, 0.0, 0.86, 0.5, -0.2]
mikata.MoveTo({jname:p for jname,p in zip(mikata.JointNames(),pose)})

#mikata.DisableTorque()
mikata.Quit()
