#!/usr/bin/python
#Enable servo motors.

from dxl_cranex7 import *

#Setup the device
crane= TCraneX7()
crane.Setup()
crane.EnableTorque()
crane.Quit()
