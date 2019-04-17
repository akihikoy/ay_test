#!/usr/bin/python
#Printing current state.

from dxl_cranex7 import *
import time

#Setup the device
crane= TCraneX7()
crane.Setup()
crane.EnableTorque()

#Relax mode:
crane.SetPWM({jname:0 for jname in crane.JointNames()})

def StateCallback(state):
  print 'State=',state

crane.StartStateObs(StateCallback)

try:
  #pose= [0, 0, 1, -1.3, 0]
  #crane.MoveTo({jname:p for jname,p in zip(crane.JointNames(),pose)})
  #pose= [p+dp for p,dp in zip(pose,[0,-0.3,0,0.3,0])]
  #crane.MoveTo({jname:p for jname,p in zip(crane.JointNames(),pose)})

  while True:
    time.sleep(0.1)
except KeyboardInterrupt:
  pass

crane.StopStateObs()

#crane.DisableTorque()
crane.Quit()
