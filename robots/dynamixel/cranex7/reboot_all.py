#!/usr/bin/python3
#Reboot Dynamixel.

from dxl_cranex7 import *

#Setup the device
crane= TCraneX7()
crane.Setup()

print('Rebooting Dynamixel...')
crane.Reboot()

#crane.DisableTorque()
crane.Quit()
