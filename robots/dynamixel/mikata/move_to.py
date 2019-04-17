#!/usr/bin/python
#Move to two target poses.

from dxl_mikata import *

#Setup the device
mikata= TMikata()
mikata.Setup()
mikata.EnableTorque()

pose= [0, 0, 1, -1.3, 0]
#pose= [0]*5
mikata.MoveTo({jname:p for jname,p in zip(mikata.JointNames(),pose)})

pose= [p+dp for p,dp in zip(pose,[0,-0.3,0,0.3,0])]
mikata.MoveTo({jname:p for jname,p in zip(mikata.JointNames(),pose)})

#mikata.DisableTorque()
mikata.Quit()
