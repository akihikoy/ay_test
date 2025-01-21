#!/usr/bin/python3
#Move to initial target position with trajectory control.

from dxl_mikata import *

#Setup the device
mikata= TMikata()
mikata.Setup()
mikata.EnableTorque()

pose= [0, 0, 0, 0, 0]
mikata.FollowTrajectory(mikata.JointNames(),[pose],[3.0],blocking=True)

#mikata.DisableTorque()
mikata.Quit()
