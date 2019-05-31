#!/usr/bin/python
#Enable servo motors.

from dxl_mikata6 import *

#Setup the device
mikata= TMikata6()
mikata.Setup()
mikata.EnableTorque()
mikata.Quit()
