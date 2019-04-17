#!/usr/bin/python
#Record key points by moving the arm.

from dxl_cranex7 import *
import time
from kbhit2 import TKBHit

#Setup the device
crane= TCraneX7()
crane.Setup()
crane.EnableTorque()

#Relax mode:
crane.SetPWM({jname:0 for jname in crane.JointNames()})

crane.StartStateObs()

try:
  kbhit= TKBHit()
  while True:
    c= kbhit.KBHit()
    if c=='q':  break
    elif c is not None:
      #print crane.State()
      print crane.State()['position']
    #crane.MoveTo(crane.Position(as_dict=True),wait=False)
    time.sleep(0.0025)
except KeyboardInterrupt:
  pass

crane.StopStateObs()

#crane.DisableTorque()
crane.Quit()
