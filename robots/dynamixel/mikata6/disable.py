#!/usr/bin/python
#Disable motors.

from dxl_mikata6 import *

#Setup the device
mikata= TMikata6()
mikata.Setup()
mikata.DisableTorque()
mikata.Quit()
