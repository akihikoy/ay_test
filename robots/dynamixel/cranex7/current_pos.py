#!/usr/bin/python3
#Printing current position without enabling torque.

from dxl_cranex7 import *
import time

#Setup the device
crane= TCraneX7()
crane.Setup()
crane.EnableTorque()

#Relax mode:
crane.SetPWM({jname:0 for jname in crane.JointNames()})

try:
  while True:
    print('Position=',crane.Position())
    time.sleep(0.001)
except KeyboardInterrupt:
  pass

#crane.DisableTorque()
crane.Quit()
