#!/usr/bin/python
#Reboot Dynamixel.

from dxl_mikata import *

#Setup the device
mikata= TMikata()
mikata.Setup()

print 'Rebooting Dynamixel...'
mikata.Reboot()

#mikata.DisableTorque()
mikata.Quit()
