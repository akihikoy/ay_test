#!/usr/bin/python3
#Printing current state.

from dxl_xw540_xw430 import *
import time

#Setup the device
device= TXW540XW430()
device.Setup()
device.EnableTorque()

#Relax mode:
device.SetPWM({jname:0 for jname in device.JointNames()})

def StateCallback(state):
  print('State=',state)

device.StartStateObs(StateCallback)

try:
  #pose= [0, 0, 1, -1.3, 0]
  #device.MoveTo({jname:p for jname,p in zip(device.JointNames(),pose)})
  #pose= [p+dp for p,dp in zip(pose,[0,-0.3,0,0.3,0])]
  #device.MoveTo({jname:p for jname,p in zip(device.JointNames(),pose)})

  while True:
    time.sleep(0.1)
except KeyboardInterrupt:
  pass

device.StopStateObs()

#device.DisableTorque()
device.Quit()
