#!/usr/bin/python3
#Relax Dynamixel servos (set all PWMs to zero).

from dxl_xw540_xw430 import *

#Setup the device
device= TXW540XW430()
device.Setup()
device.EnableTorque()

device.SetPWM({jname:0 for jname in device.JointNames()})

#device.DisableTorque()
device.Quit()
