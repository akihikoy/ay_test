#!/usr/bin/python3
#Relax Dynamixel servos (set all PWMs to zero).

from dxl_mikata6 import *

#Setup the device
mikata= TMikata6()
mikata.Setup()
mikata.EnableTorque()

mikata.SetPWM({jname:0 for jname in mikata.JointNames()})

#mikata.DisableTorque()
mikata.Quit()
