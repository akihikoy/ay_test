#!/usr/bin/python3
#Move to two target poses.

from dxl_xw540_xw430 import *
import numpy as np

#Setup the device
device= TXW540XW430()
device.Setup()
device.EnableTorque()

pose= [ -0.7578, -1.3744 ]
#pose= [0]*5
device.MoveTo({jname:p for jname,p in zip(device.JointNames(),pose)})

#pose= np.array(pose)+[0,0,-0.6,-0.6]
#device.MoveTo({jname:p for jname,p in zip(device.JointNames(),pose)})

#device.DisableTorque()
device.Quit()
