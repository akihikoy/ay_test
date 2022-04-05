#!/usr/bin/python
#Reboot Dynamixel.

from dxl_fd2f4dof import *

#Setup the device
gripper= TFD2F4DoF()
gripper.Setup()

print 'Rebooting Dynamixel...'
gripper.Reboot()

#gripper.DisableTorque()
gripper.Quit()
