#!/usr/bin/python3
#Printing current position without enabling torque.

from dxl_xw540_xw430 import *
import time

#Setup the device
device= TXW540XW430()
device.Setup()
device.EnableTorque()

#Relax mode:
#device.SetPWM({jname:0 for jname in device.JointNames()})
device.DisableTorque()

try:
  while True:
    print('Position=','[',', '.join(['{:.4f}'.format(p) for p in device.Position()]),']')
    time.sleep(0.001)
except KeyboardInterrupt:
  pass

#device.DisableTorque()
device.Quit()
