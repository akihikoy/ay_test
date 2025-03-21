#!/usr/bin/python3
#Reboot Dynamixel.

from dxl_xw540_xw430 import *

#Setup the device
device= TXW540XW430()
device.Setup()

print('Rebooting Dynamixel...')
device.Reboot()

#gripper.DisableTorque()
device.Quit()
