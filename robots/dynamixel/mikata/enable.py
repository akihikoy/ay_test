#!/usr/bin/python3
#Enable servo motors.

from dxl_mikata import *

#Setup the device
mikata= TMikata()
mikata.Setup()
mikata.EnableTorque()
mikata.Quit()
