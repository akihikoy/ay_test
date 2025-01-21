#!/usr/bin/python3
#Disable motors.

from dxl_fd2f4dof import *

#Setup the device
gripper= TFD2F4DoF()
gripper.Setup()
gripper.DisableTorque()
gripper.Quit()
