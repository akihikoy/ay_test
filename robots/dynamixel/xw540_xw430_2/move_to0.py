#!/usr/bin/python3
#Move to initial target position with trajectory control.

from dxl_xw540_xw430 import *

#Setup the device
device= TXW540XW430()
device.Setup()
device.EnableTorque()

pose= [ -0.7578, -1.3744 ]
device.FollowTrajectory(device.JointNames(),[pose],[3.0],blocking=True)

#device.DisableTorque()
device.Quit()
