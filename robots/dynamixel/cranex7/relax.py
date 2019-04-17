#!/usr/bin/python
#Relax Dynamixel servos (set all PWMs to zero).

from dxl_cranex7 import *

#Setup the device
crane= TCraneX7()
crane.Setup()
crane.EnableTorque()

crane.SetPWM({jname:0 for jname in crane.JointNames()})

#crane.DisableTorque()
crane.Quit()
