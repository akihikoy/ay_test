#!/usr/bin/python
#Printing current position without enabling torque.

from dxl_fd2f4dof import *
import time

#Setup the device
gripper= TFD2F4DoF()
gripper.Setup()
gripper.EnableTorque()

#Relax mode:
#gripper.SetPWM({jname:0 for jname in gripper.JointNames()})
gripper.DisableTorque()

try:
  while True:
    print 'Position=','[',', '.join(['{:.4f}'.format(p) for p in gripper.Position()]),']'
    time.sleep(0.001)
except KeyboardInterrupt:
  pass

#gripper.DisableTorque()
gripper.Quit()
