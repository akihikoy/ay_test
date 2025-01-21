#!/usr/bin/python3
#Disable motors.

from dxl_mikata import *

#Setup the device
mikata= TMikata()
mikata.Setup()
mikata.DisableTorque()
mikata.Quit()
