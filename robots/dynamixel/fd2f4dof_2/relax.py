#!/usr/bin/python
#Relax Dynamixel servos (set all PWMs to zero).

from dxl_fd2f4dof import *

#Setup the device
gripper= TFD2F4DoF()
gripper.Setup()
gripper.EnableTorque()

gripper.SetPWM({jname:0 for jname in gripper.JointNames()})

#gripper.DisableTorque()
gripper.Quit()
