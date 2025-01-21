#!/usr/bin/python3
#Move to initial target position with trajectory control.

from dxl_mikata6 import *

#Setup the device
mikata= TMikata6()
mikata.Setup()
mikata.EnableTorque()

pose= [0.0]*7
mikata.FollowTrajectory(mikata.JointNames(),[pose],[8.0],blocking=True)

#mikata.DisableTorque()
mikata.Quit()
