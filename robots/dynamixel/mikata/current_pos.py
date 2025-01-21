#!/usr/bin/python3
#Printing current position without enabling torque.

from dxl_mikata import *
import time

#Setup the device
mikata= TMikata()
mikata.Setup()
mikata.EnableTorque()

#Relax mode:
mikata.SetPWM({jname:0 for jname in mikata.JointNames()})

try:
  while True:
    print('Position=',mikata.Position())
    time.sleep(0.001)
except KeyboardInterrupt:
  pass

#mikata.DisableTorque()
mikata.Quit()
