#!/usr/bin/python3
#Reboot Dynamixel.

from dxl_mikata6 import *

#Setup the device
mikata= TMikata6()
mikata.Setup()

print('Rebooting Dynamixel...')
mikata.Reboot()

#mikata.DisableTorque()
mikata.Quit()
