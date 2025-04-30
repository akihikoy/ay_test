#!/usr/bin/python3
#Disable motors.

from dxl_xw540_xw430 import *

#Setup the device
device= TXW540XW430()
device.Setup()
device.DisableTorque()
device.Quit()
