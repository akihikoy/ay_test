#!/usr/bin/python
#Move to initial target position with trajectory control.

from dxl_cranex7 import *

#Setup the device
crane= TCraneX7()
crane.Setup()
crane.EnableTorque()

pose= [0.0]*8
crane.FollowTrajectory(crane.JointNames(),[pose],[8.0],blocking=True)

#crane.DisableTorque()
crane.Quit()
