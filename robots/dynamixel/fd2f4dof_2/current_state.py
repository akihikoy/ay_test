#!/usr/bin/python3
#Printing current state.

from dxl_fd2f4dof import *
import time

#Setup the device
gripper= TFD2F4DoF()
gripper.Setup()
gripper.EnableTorque()

#Relax mode:
gripper.SetPWM({jname:0 for jname in gripper.JointNames()})

def StateCallback(state):
  print('State=',state)

gripper.StartStateObs(StateCallback)

try:
  #pose= [0, 0, 1, -1.3, 0]
  #gripper.MoveTo({jname:p for jname,p in zip(gripper.JointNames(),pose)})
  #pose= [p+dp for p,dp in zip(pose,[0,-0.3,0,0.3,0])]
  #gripper.MoveTo({jname:p for jname,p in zip(gripper.JointNames(),pose)})

  while True:
    time.sleep(0.1)
except KeyboardInterrupt:
  pass

gripper.StopStateObs()

#gripper.DisableTorque()
gripper.Quit()
