#!/usr/bin/python
#Disable motors.

from dxl_mikata import *

#Setup the device
mikata= TMikata()
mikata.Setup()
mikata.DisableTorque()
mikata.Quit()
