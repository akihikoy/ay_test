#!/usr/bin/python3
#Following sine curve.

from dxl_mikata import *
import time
import math
import numpy as np

#Setup the device
mikata= TMikata()
mikata.Setup()
mikata.EnableTorque()

#Move to initial pose
p_start= [0, 0, 1, -1.3, 0]
mikata.MoveTo({jname:p for jname,p in zip(mikata.JointNames(),p_start)})
time.sleep(0.5)
print('Current position=',mikata.Position())

#Move to a target position
gain= [0.45, 0.15, 0.15, 0.7, 0.7]
angvel= [1, 2, 1, 3, 2]
for t in np.mgrid[0:2*math.pi:0.05]:
  p_trg= [p0 + g*math.sin(w*t) for p0,g,w in zip(p_start,gain,angvel)]
  #print p_trg
  mikata.MoveTo({jname:p for jname,p in zip(mikata.JointNames(),p_trg)}, blocking=False)
  time.sleep(0.025)
  #print 'Current position=',dxl.Position()


#mikata.DisableTorque()
mikata.Quit()

