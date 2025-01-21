#!/usr/bin/python3
#Printing current state.

from dxl_mikata6 import *
import time

#Setup the device
mikata= TMikata6()
mikata.Setup()
mikata.EnableTorque()

#Relax mode:
mikata.SetPWM({jname:0 for jname in mikata.JointNames()})

def StateCallback(state):
  print('State=',state)

mikata.StartStateObs(StateCallback)

try:
  #pose= [0, 0, 1, -1.3, 0]
  #mikata.MoveTo({jname:p for jname,p in zip(mikata.JointNames(),pose)})
  #pose= [p+dp for p,dp in zip(pose,[0,-0.3,0,0.3,0])]
  #mikata.MoveTo({jname:p for jname,p in zip(mikata.JointNames(),pose)})

  while True:
    time.sleep(0.1)
except KeyboardInterrupt:
  pass

mikata.StopStateObs()

#mikata.DisableTorque()
mikata.Quit()
